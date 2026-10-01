from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.generator import AnswerGenerator
from app.retriever import Retriever


def _docs():
    return [
        Document(page_content="BRCA1 mutations increase breast cancer risk", metadata={"source": "a.pdf", "page": 1, "topic": "Cancer"}),
        Document(page_content="Statins lower cholesterol and heart disease risk", metadata={"source": "b.pdf", "page": 2, "topic": "Cardiovascular"}),
    ]


def test_search_returns_best_match_first():
    r = Retriever(DeterministicFakeEmbedding(size=64))
    r.add(_docs())
    top = r.search("BRCA1 mutations increase breast cancer risk", k=1)
    assert top[0].metadata["source"] == "a.pdf"


def test_topic_filter_restricts_results():
    r = Retriever(DeterministicFakeEmbedding(size=64))
    r.add(_docs())
    results = r.search("anything at all", k=5, topic="Cardiovascular")
    assert results and all(d.metadata["topic"] == "Cardiovascular" for d in results)


def test_empty_store_returns_nothing():
    assert Retriever(DeterministicFakeEmbedding(size=64)).search("q") == []


def test_index_persists_to_disk(tmp_path):
    emb = DeterministicFakeEmbedding(size=64)
    Retriever(emb, tmp_path).add(_docs())
    reloaded = Retriever(emb, tmp_path)
    assert len(reloaded.search("x", k=5)) == 2


class CapturingLLM:
    def __init__(self):
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return type("Resp", (), {"content": "Answer [1]"})()


def test_generator_builds_grounded_prompt():
    llm = CapturingLLM()
    answer = AnswerGenerator(llm).generate("What raises breast cancer risk?", _docs())
    assert answer == "Answer [1]"
    assert "ONLY" in llm.prompt and "[1] (a.pdf, p.1)" in llm.prompt and "What raises breast cancer risk?" in llm.prompt
