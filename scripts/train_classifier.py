"""Train the topic classifier on the CSV from fetch_pubmed.py and print evaluation metrics.

Usage:  python scripts/train_classifier.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.classifier import TopicClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/pubmed_abstracts.csv")
    ap.add_argument("--out", default=str(config.CLASSIFIER_PATH))
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["abstract", "topic"])
    print(f"Rows: {len(df)} | class counts:\n{df['topic'].value_counts()}\n")

    clf = TopicClassifier()
    m = clf.train(df["abstract"].tolist(), df["topic"].tolist())
    print(f"Accuracy: {m['accuracy']:.3f} | Macro F1: {m['f1_macro']:.3f}\n")
    print(m["report"])
    print("Confusion matrix (rows = true, cols = predicted):", m["labels"])
    for row in m["confusion_matrix"]:
        print(row)
    clf.save(Path(args.out))
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
