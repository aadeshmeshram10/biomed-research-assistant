"""Central configuration. Values come from environment variables / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "data" / "index"))
CLASSIFIER_PATH = Path(os.getenv("CLASSIFIER_PATH", BASE_DIR / "models" / "topic_classifier.joblib"))

MAX_UPLOAD_MB = 20
