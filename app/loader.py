"""Step 1 - Ingest: read a PDF and return cleaned text, one string per page."""
import io
import re

from pypdf import PdfReader


class DocumentLoader:
    def load_pdf_bytes(self, data: bytes) -> list[str]:
        """Return a list with one cleaned string per page (empty pages stay as '')
        so that list index + 1 always equals the real page number."""
        reader = PdfReader(io.BytesIO(data))
        return [self.clean_text(page.extract_text() or "") for page in reader.pages]

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"-\n(?=\w)", "", text)  # re-join words hyphenated across lines
        text = re.sub(r"\s+", " ", text)       # collapse newlines/tabs/multiple spaces
        return text.strip()
