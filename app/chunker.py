"""Step 1 - Ingest: split page text into overlapping chunks that carry metadata."""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, pages: list[str], source: str, topic: str = "Unclassified") -> list[Document]:
        docs: list[Document] = []
        for page_no, text in enumerate(pages, start=1):
            if not text:
                continue
            for i, chunk in enumerate(self.splitter.split_text(text)):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": source, "page": page_no, "topic": topic, "chunk": i},
                    )
                )
        return docs
