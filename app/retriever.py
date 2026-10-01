"""Step 3 - RAG retrieval: embed chunks, store them in FAISS, search by meaning."""
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class Retriever:
    def __init__(self, embeddings, index_dir: Optional[Path] = None):
        self.embeddings = embeddings
        self.index_dir = Path(index_dir) if index_dir else None
        self.store: Optional[FAISS] = None
        if self.index_dir and (self.index_dir / "index.faiss").exists():
            self.store = FAISS.load_local(
                str(self.index_dir), embeddings, allow_dangerous_deserialization=True
            )

    def add(self, docs: list[Document]) -> None:
        if not docs:
            return
        if self.store is None:
            self.store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.store.add_documents(docs)
        if self.index_dir:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.store.save_local(str(self.index_dir))

    def search(self, query: str, k: int = 4, topic: Optional[str] = None) -> list[Document]:
        if self.store is None:
            return []
        metadata_filter = {"topic": topic} if topic else None
        return self.store.similarity_search(
            query, k=k, filter=metadata_filter, fetch_k=max(20, k * 5)
        )
