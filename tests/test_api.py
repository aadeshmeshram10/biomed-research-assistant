import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.classifier import TopicClassifier
from app.generator import AnswerGenerator
from app.loader import DocumentLoader
from app.main import create_app
from app.retriever import Retriever


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        DocumentLoader, "load_pdf_bytes",
        lambda self, data: ["BRCA1 mutations raise breast cancer risk in carriers."],
    )
    app = create_app(
        retriever=Retriever(DeterministicFakeEmbedding(size=64)),
        generator=AnswerGenerator(FakeListChatModel(responses=["BRCA1 raises risk [1]."])),
        classifier=TopicClassifier(),  # untrained -> 'Unclassified'
    )
    with TestClient(app) as c:
        yield c


def _upload(client, name="paper.pdf"):
    return client.post("/ingest", files={"file": (name, b"%PDF-fake", "application/pdf")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "papers": 0}


def test_ingest_then_list_then_ask(client):
    r = _upload(client)
    assert r.status_code == 200 and r.json()["chunks"] >= 1
    assert "paper.pdf" in client.get("/papers").json()

    r = client.post("/ask", json={"question": "What does BRCA1 do?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"] == "BRCA1 raises risk [1]."
    assert body["sources"][0]["source"] == "paper.pdf" and body["sources"][0]["page"] == 1


def test_rejects_non_pdf(client):
    r = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_duplicate_ingest_conflicts(client):
    _upload(client)
    assert _upload(client).status_code == 409


def test_ask_before_ingest_is_400(client):
    assert client.post("/ask", json={"question": "anything here"}).status_code == 400


def test_ask_validates_input(client):
    _upload(client)
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "k": 99}).status_code == 422


def test_quota_error_becomes_friendly_429(monkeypatch):
    class QuotaLLM:
        def invoke(self, prompt):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    monkeypatch.setattr(DocumentLoader, "load_pdf_bytes", lambda self, data: ["Some text about genes."])
    app = create_app(
        retriever=Retriever(DeterministicFakeEmbedding(size=64)),
        generator=AnswerGenerator(QuotaLLM()),
        classifier=TopicClassifier(),
    )
    with TestClient(app) as c:
        c.post("/ingest", files={"file": ("p.pdf", b"%PDF-fake", "application/pdf")})
        r = c.post("/ask", json={"question": "what about genes?"})
        assert r.status_code == 429 and "free-tier" in r.json()["detail"]


def test_frontend_is_served(client):
    r = client.get("/ui/")
    assert r.status_code == 200 and "BioMed Research Assistant" in r.text
