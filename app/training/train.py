from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from joblib import dump
from pathlib import Path
from app.utils.data_loader import load_dataset
from app.preprocessing.text_cleaner import clean_text, clean_batch

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

X_train, y_train, X_test, y_test = load_dataset()
X_train = clean_batch(X_train)
X_test = clean_batch(X_test)

pipeline = Pipeline([
    (
        "tfidf", TfidfVectorizer(max_features=5000)
    ),
    (
        "classifier", LogisticRegression(max_iter=1000)
    )
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(accuracy)
print(classification_report(y_test, predictions))

MODEL_DIR = MODEL_DIR / "developer_intent_classifier.joblib"
dump(pipeline, MODEL_DIR)