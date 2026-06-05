from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import (
confusion_matrix, accuracy_score, classification_report
)
from app.utils.data_loader import load_dataset
from app.preprocessing.text_cleaner import clean_batch
from sklearn.metrics import f1_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / "developer_intent_classifier.joblib"

pipeline = load(MODEL_PATH)

X_train, y_train, X_test, y_test = load_dataset()
X_test = clean_batch(X_test)

predictions = pipeline.predict(X_test)
print("Classification report: ",classification_report(y_test, predictions, zero_division=0))
print("Accuracy: ",accuracy_score(y_test, predictions))
print("F1 score: ",f1_score(y_test, predictions, average="macro"))
print("Weighted F1 score: ", f1_score(y_test, predictions, average="weighted"))

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize = (20, 20))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()