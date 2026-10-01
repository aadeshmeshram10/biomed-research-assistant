"""Step 2 - ML tagging: TF-IDF + Logistic Regression topic classifier (scikit-learn)."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

UNCLASSIFIED = "Unclassified"


class TopicClassifier:
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold  # below this confidence we refuse to guess
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        self.trained = False

    def train(self, texts, labels, test_size: float = 0.2, seed: int = 42) -> dict:
        """Fit on a train split, evaluate on a held-out test split, return metrics."""
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=seed
        )
        self.pipeline.fit(X_train, y_train)
        self.trained = True
        preds = self.pipeline.predict(X_test)
        label_names = sorted(set(labels))
        return {
            "accuracy": accuracy_score(y_test, preds),
            "f1_macro": f1_score(y_test, preds, average="macro"),
            "labels": label_names,
            "confusion_matrix": confusion_matrix(y_test, preds, labels=label_names).tolist(),
            "report": classification_report(y_test, preds),
        }

    def predict(self, text: str) -> tuple[str, float]:
        """Return (topic, confidence). Falls back to 'Unclassified' if unsure."""
        if not self.trained:
            return UNCLASSIFIED, 0.0
        probs = self.pipeline.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = str(self.pipeline.named_steps["clf"].classes_[idx])
        return (label, confidence) if confidence >= self.threshold else (UNCLASSIFIED, confidence)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load_or_empty(cls, path: Path) -> "TopicClassifier":
        """Load a trained model if the file exists, else return an untrained one."""
        obj = cls()
        if Path(path).exists():
            obj.pipeline = joblib.load(path)
            obj.trained = True
        return obj
