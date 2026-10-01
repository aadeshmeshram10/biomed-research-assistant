import pytest

from app.chunker import Chunker


def test_split_keeps_metadata_and_page_numbers():
    pages = ["A" * 50, "", "B" * 50]  # page 2 is empty
    docs = Chunker(chunk_size=100, chunk_overlap=10).split(pages, source="x.pdf", topic="Cancer")
    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert all(d.metadata["source"] == "x.pdf" and d.metadata["topic"] == "Cancer" for d in docs)


def test_long_page_is_split_into_multiple_chunks():
    text = " ".join(["gene"] * 500)
    docs = Chunker(chunk_size=200, chunk_overlap=20).split([text], source="x.pdf")
    assert len(docs) > 1
    assert all(len(d.page_content) <= 200 for d in docs)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        Chunker(chunk_size=100, chunk_overlap=100)
