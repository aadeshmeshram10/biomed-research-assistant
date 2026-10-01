"""Build a labelled dataset of PubMed abstracts (topic = the MeSH search used).

Usage:  python scripts/fetch_pubmed.py --per-topic 300
Output: data/pubmed_abstracts.csv  (columns: topic, pmid, abstract)
"""
import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOPIC_QUERIES = {
    "Genetics": '"genetics"[MeSH Major Topic]',
    "Cancer": '"neoplasms"[MeSH Major Topic]',
    "Infectious Disease": '"communicable diseases"[MeSH Major Topic]',
    "Neuroscience": '"nervous system diseases"[MeSH Major Topic]',
    "Cardiovascular": '"cardiovascular diseases"[MeSH Major Topic]',
}


def search_ids(query: str, n: int) -> list[str]:
    params = {
        "db": "pubmed",
        "term": f"{query} AND hasabstract AND 2020:2026[dp]",
        "retmax": n,
        "retmode": "json",
    }
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def fetch_abstracts(ids: list[str]) -> list[tuple[str, str]]:
    r = requests.post(
        f"{EUTILS}/efetch.fcgi",
        data={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        timeout=60,
    )
    r.raise_for_status()
    out = []
    for art in ET.fromstring(r.content).findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID")
        parts = ["".join(t.itertext()) for t in art.findall(".//Abstract/AbstractText")]
        if pmid and parts:
            out.append((pmid, " ".join(parts)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=300)
    ap.add_argument("--out", default="data/pubmed_abstracts.csv")
    args = ap.parse_args()

    rows = []
    for topic, query in TOPIC_QUERIES.items():
        ids = search_ids(query, args.per_topic)
        for i in range(0, len(ids), 100):
            for pmid, abstract in fetch_abstracts(ids[i : i + 100]):
                rows.append({"topic": topic, "pmid": pmid, "abstract": abstract})
            time.sleep(0.4)  # stay under NCBI's 3 requests/second limit
        print(f"{topic}: fetched {len(ids)} ids")

    df = pd.DataFrame(rows).dropna()
    df = df[~df.duplicated(subset="pmid", keep=False)]  # drop papers matched by 2+ topics (ambiguous label)
    df = df[df["abstract"].str.len() > 200]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df["topic"].value_counts())


if __name__ == "__main__":
    main()
