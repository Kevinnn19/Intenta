# scripts/compute_classification_metrics.py
import json
from pathlib import Path
from collections import Counter
import joblib
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "developer_intent_classifier.joblib"
DATASET_PATH = ROOT / "datasets" / "developer_intents.json"

def load_dataset(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    texts = []
    labels = []
    # format 1: dict intent -> list of examples
    if isinstance(data, dict):
        for intent, examples in data.items():
            if isinstance(examples, list):
                for ex in examples:
                    if isinstance(ex, dict):
                        # try expected fields
                        txt = ex.get("text") or ex.get("utterance") or ex.get("example") or ex.get("query")
                    else:
                        txt = str(ex)
                    texts.append(txt)
                    labels.append(intent)
    # format 2: list of {text, intent}
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("utterance") or item.get("query") or item.get("example")
                intent = item.get("intent") or item.get("label") or item.get("intent_label")
                if txt is None or intent is None:
                    # try flattening structure
                    for v in item.values():
                        if isinstance(v, str) and len(txt or "") < len(v):
                            txt = v
                if txt is None or intent is None:
                    continue
                texts.append(txt)
                labels.append(intent)
    else:
        raise RuntimeError("Unknown dataset format in " + str(path))

    return texts, labels

def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Train first: python app/training/train.py")
        return
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}")
        return

    print("Loading dataset...")
    X, y = load_dataset(DATASET_PATH)
    print(f"Loaded {len(X)} examples across {len(set(y))} intent(s).")

    print("Loading model...")
    clf = joblib.load(MODEL_PATH)

    # Predict
    print("Predicting...")
    # If pipeline supports predict or predict_proba
    try:
        y_pred = clf.predict(X)
    except Exception as e:
        print("clf.predict failed:", e)
        # attempt to use pipeline steps
        raise

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, digits=4)
    counts = Counter(y)
    print(f"\nAccuracy: {acc*100:.2f}%")
    print("\nClassification report:\n")
    print(report)
    print("\nPer-intent counts (top 10):")
    for intent, cnt in counts.most_common(10):
        print(f"  {intent}: {cnt}")

if __name__ == "__main__":
    main()