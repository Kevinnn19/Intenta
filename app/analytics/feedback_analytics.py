import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent.parent

FEEDBACK_PATH = BASE_DIR / "datasets" / "feedback" / "feedback_data.json"

def analyze_feedback():
    if not FEEDBACK_PATH.exists():
        return {
            "message": "No feedback data found"
        }

    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_feedback = len(data)

    if total_feedback == 0:
        return {
            "message": "No feedback records available"
        }

    positive = sum(1 for item in data if item["feedback"] == "positive")
    negative = sum(1 for item in data if item["feedback"] == "negative")
    positive_rate = round(positive / total_feedback * 100, 2)
    negative_rate = round(negative / total_feedback * 100, 2)

    failed_intents = [item["predicted_intent"] for item in data if item["feedback"] == "negative"]
    successful_intents = [item["predicted_intent"] for item in data if item["feedback"] == "positive"]

    most_failed_intent = None
    most_successful_intent = None

    if failed_intents:
        most_failed_intent = (
            Counter(failed_intents).most_common(1)[0][0]
        )

    if successful_intents:
        most_successful_intent = (
            Counter(successful_intents).most_common(1)[0][0]
        )

    return {
        "total_feedback": total_feedback,
        "positive_feedback": positive,
        "negative_feedback": negative,
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "most_failed_intent": most_failed_intent,
        "most_successful_intent": most_successful_intent
    }