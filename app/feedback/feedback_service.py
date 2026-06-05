import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent

FEEDBACK_PATH = BASE_DIR / "datasets" / "feedback" / "feedback_data.json"

class FeedbackService:
    def save_feedback(self, query, predicted_intent, source, feedback):

        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not FEEDBACK_PATH.exists():
            with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
                json.dump([], f)

        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "predicted_intent": predicted_intent,
            "source": source,
            "feedback": feedback
        })

        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

feedback_service = FeedbackService()