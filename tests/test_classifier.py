from app.classifier import UNCLASSIFIED, TopicClassifier

# Tiny hand-written fixture, only to test the code path (real training uses PubMed data).
DATA = {
    "Cancer": [
        "tumor cells metastasis chemotherapy oncology carcinoma",
        "malignant neoplasm tumor growth radiotherapy patients",
        "breast cancer tumor suppressor chemotherapy resistance",
        "leukemia lymphoma cancer cells treatment oncology",
        "carcinoma metastasis tumor biopsy oncology outcomes",
        "cancer immunotherapy tumor microenvironment response",
    ],
    "Cardiovascular": [
        "heart failure myocardial infarction cardiac patients",
        "hypertension blood pressure cardiovascular risk stroke",
        "coronary artery disease cardiac surgery outcomes",
        "atrial fibrillation heart rhythm anticoagulant therapy",
        "cardiac arrest heart ventricular function study",
        "cholesterol atherosclerosis cardiovascular disease risk",
    ],
    "Infectious Disease": [
        "virus infection antibiotic resistance bacteria outbreak",
        "viral pathogen transmission vaccine infection rates",
        "bacterial infection antimicrobial treatment sepsis",
        "malaria tuberculosis infection epidemic control",
        "hiv infection antiviral therapy pathogen spread",
        "influenza virus vaccine outbreak surveillance infection",
    ],
}


def _xy():
    texts = [t for ts in DATA.values() for t in ts]
    labels = [label for label, ts in DATA.items() for _ in ts]
    return texts, labels


def test_train_returns_metrics_and_predicts():
    texts, labels = _xy()
    clf = TopicClassifier(threshold=0.0)
    metrics = clf.train(texts, labels, test_size=0.34)
    assert {"accuracy", "f1_macro", "confusion_matrix", "report"} <= set(metrics)
    label, conf = clf.predict("tumor chemotherapy oncology cancer cells")
    assert label == "Cancer" and 0 < conf <= 1


def test_untrained_returns_unclassified():
    assert TopicClassifier().predict("anything") == (UNCLASSIFIED, 0.0)


def test_save_and_load_roundtrip(tmp_path):
    texts, labels = _xy()
    clf = TopicClassifier(threshold=0.0)
    clf.train(texts, labels, test_size=0.34)
    clf.save(tmp_path / "m.joblib")
    loaded = TopicClassifier.load_or_empty(tmp_path / "m.joblib")
    loaded.threshold = 0.0
    assert loaded.trained
    assert loaded.predict("heart cardiac blood pressure")[0] == "Cardiovascular"


def test_low_confidence_falls_back_to_unclassified():
    texts, labels = _xy()
    clf = TopicClassifier(threshold=0.99)
    clf.train(texts, labels, test_size=0.34)
    assert clf.predict("completely unrelated words here")[0] == UNCLASSIFIED
