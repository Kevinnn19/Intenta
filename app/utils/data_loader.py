import json
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR / "datasets" / "developer_intents.json"


def load_dataset():

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []

    # Convert JSON structure
    for intent, examples in data.items():

        for text in examples:

            texts.append(text)
            labels.append(intent)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    return X_train, y_train, X_test, y_test