"""Step 3 - RAG generation: a grounded, citation-forcing prompt sent to the LLM."""
from langchain_core.documents import Document

PROMPT_TEMPLATE = """You are a biomedical research assistant.
Answer the question using ONLY the numbered context passages below.
Rules:
- Cite the passages you used like [1] or [2][3] right after each claim.
- If the context does not contain the answer, say "I could not find this in the provided papers."
- Do not use outside knowledge. Do not invent facts or citations.

Context:
{context}

Question: {question}

Answer:"""


class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def build_context(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[{i}] ({d.metadata.get('source')}, p.{d.metadata.get('page')}) {d.page_content}"
            for i, d in enumerate(docs, start=1)
        )

    def generate(self, question: str, docs: list[Document]) -> str:
        prompt = PROMPT_TEMPLATE.format(context=self.build_context(docs), question=question)
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
