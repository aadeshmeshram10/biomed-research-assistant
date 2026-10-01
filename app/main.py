"""Step 4 - REST API (FastAPI). Wires loader -> classifier -> chunker -> retriever -> generator."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config
from app.chunker import Chunker
from app.classifier import TopicClassifier
from app.generator import AnswerGenerator
from app.loader import DocumentLoader
from app.retriever import Retriever


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    topic: Optional[str] = Field(None, description="Only search papers tagged with this topic")
    k: int = Field(4, ge=1, le=10, description="Number of passages to retrieve")


def _build_defaults():
    """Heavy imports live here so tests can inject fakes without downloading models."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL, google_api_key=config.GOOGLE_API_KEY, temperature=0
    )
    return Retriever(embeddings, config.INDEX_DIR), AnswerGenerator(llm)


def create_app(retriever=None, generator=None, classifier=None) -> FastAPI:
    loader, chunker = DocumentLoader(), Chunker()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        r, g = retriever, generator
        if r is None or g is None:
            default_r, default_g = _build_defaults()
            r, g = r or default_r, g or default_g
        app.state.retriever, app.state.generator = r, g
        app.state.classifier = classifier or TopicClassifier.load_or_empty(config.CLASSIFIER_PATH)
        app.state.registry_path = (r.index_dir / "papers.json") if r.index_dir else None
        path = app.state.registry_path
        app.state.papers = json.loads(path.read_text()) if path and path.exists() else {}
        yield

    app = FastAPI(title="BioMed Research Assistant", lifespan=lifespan)

    def _save_registry():
        path: Optional[Path] = app.state.registry_path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(app.state.papers, indent=2))

    @app.get("/health")
    def health():
        return {"status": "ok", "papers": len(app.state.papers)}

    @app.post("/ingest")
    async def ingest(file: UploadFile = File(...)):
        name = os.path.basename(file.filename or "")
        if not name.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are supported.")
        data = await file.read()
        if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB.")
        if name in app.state.papers:
            raise HTTPException(409, f"'{name}' was already ingested.")
        try:
            pages = loader.load_pdf_bytes(data)
        except Exception:
            raise HTTPException(422, "Could not read this PDF.")
        full_text = " ".join(pages).strip()
        if not full_text:
            raise HTTPException(422, "No extractable text found (is it a scanned PDF?).")

        topic, confidence = app.state.classifier.predict(full_text[:3000])
        chunks = chunker.split(pages, source=name, topic=topic)
        app.state.retriever.add(chunks)
        record = {
            "topic": topic,
            "confidence": round(confidence, 3),
            "pages": len(pages),
            "chunks": len(chunks),
        }
        app.state.papers[name] = record
        _save_registry()
        return {"source": name, **record}

    @app.get("/papers")
    def papers():
        return app.state.papers

    @app.post("/ask")
    def ask(req: AskRequest):
        if not app.state.papers:
            raise HTTPException(400, "No papers ingested yet. Upload a PDF to /ingest first.")
        docs = app.state.retriever.search(req.question, k=req.k, topic=req.topic)
        if not docs:
            return {"answer": "No relevant passages found.", "sources": []}
        try:
            answer = app.state.generator.generate(req.question, docs)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                raise HTTPException(
                    429,
                    "The Gemini free-tier limit is used up. Wait a while (daily limits reset) "
                    "or set a different GEMINI_MODEL in .env and restart.",
                )
            raise HTTPException(502, f"LLM call failed: {msg}")
        sources = [
            {
                "id": i,
                "source": d.metadata["source"],
                "page": d.metadata["page"],
                "topic": d.metadata["topic"],
                "snippet": d.page_content[:200],
            }
            for i, d in enumerate(docs, start=1)
        ]
        return {"answer": answer, "sources": sources}

    frontend_dir = config.BASE_DIR / "frontend"
    if (frontend_dir / "index.html").exists():
        app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def home():
            return RedirectResponse("/ui/")

    return app


app = create_app()
