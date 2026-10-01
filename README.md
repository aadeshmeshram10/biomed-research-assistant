# BioMed Research Assistant

Upload biomedical papers (PDF) and ask questions. Answers are generated **only** from your papers and come with numbered citations (file + page). Each paper is also auto-tagged by topic with a scikit-learn classifier, and questions can be filtered by that topic.

## Architecture

```
PDF -> DocumentLoader -> TopicClassifier -> Chunker -> Retriever (FAISS)
                                                           |
Question ------------------------------------------------> top-k passages -> AnswerGenerator (LLM) -> cited answer
```

| Component | File | What it does |
|---|---|---|
| DocumentLoader | `app/loader.py` | Extracts and cleans text per page (pypdf) |
| TopicClassifier | `app/classifier.py` | TF-IDF + Logistic Regression topic tagger |
| Chunker | `app/chunker.py` | Splits pages into overlapping chunks with metadata |
| Retriever | `app/retriever.py` | Sentence-transformer embeddings + FAISS search, topic filter |
| AnswerGenerator | `app/generator.py` | Grounded prompt -> Gemini, citations required |
| API | `app/main.py` | FastAPI: `/ingest`, `/ask`, `/papers`, `/health` |

## Setup

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add your free Gemini key from aistudio.google.com/apikey
```

### (Optional) Train the topic classifier
```bash
python scripts/fetch_pubmed.py --per-topic 300        # downloads labelled abstracts from PubMed
python scripts/train_classifier.py                    # prints accuracy, F1, confusion matrix; saves model
```
Without a trained model, papers are tagged `Unclassified` and everything else still works.

### Run
```bash
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

### Test
```bash
pytest
```

## API

- `POST /ingest` - upload a PDF (`multipart/form-data`, field `file`)
- `POST /ask` - `{"question": "...", "topic": "Cancer" (optional), "k": 4}`
- `GET /papers` - list ingested papers with topic, pages, chunks
- `GET /health`

## Results
Add your classifier metrics here after training (accuracy, macro F1, confusion matrix).
