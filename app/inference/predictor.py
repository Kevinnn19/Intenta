import time
from pathlib import Path
from joblib import load
from app.routing.intent_router import IntentRouter
from app.preprocessing.text_cleaner import clean_text
from app.retrieval.retrieval_manager import RetrievalManager
from app.language.language_service import language_service

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / "developer_intent_classifier.joblib"

pipeline = load(MODEL_PATH)
router = IntentRouter()
retrieval_manager = RetrievalManager()

def predict_intent(text: str):
    start_time = time.time()

    language_info = language_service.process_query(text)
    processed_text = language_info["processed_text"]
    cleaned_text = clean_text(processed_text)

    probabilities = pipeline.predict_proba([cleaned_text])[0]

    labels = pipeline.classes_

    intent_score = list(zip(labels, probabilities))
    intent_score = sorted(intent_score, key=lambda x: x[1], reverse=True)

    top_intent, top_score = intent_score[0]
    second_intent, second_score = intent_score[1]
    confidence_gap = top_score - second_score

    if top_score < 0.05 or confidence_gap < 0.01:
        top_intent = "uncertain"
        source = "general"
    else:
        source = router.route(top_intent)

    retrieved_docs = retrieval_manager.retrieve(
        query=cleaned_text,
        source=source,
        intent=top_intent
    )

    alternatives = [
        {
            "intent": str(intent),
            "score": round(float(score), 4)
        }
        for intent, score in intent_score[1:4]
    ]
    latency_ms = round((time.time() - start_time) * 1000, 2)
    return {
        "prediction": str(top_intent),
        "confidence": round(float(top_score), 4),
        "language": language_info["language"],
        "source": source,
        "retrieved_docs": retrieved_docs,
        "alternatives": alternatives,
        "latency_ms": latency_ms
    }