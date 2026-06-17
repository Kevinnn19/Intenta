# scripts/count_non_english_feedback.py
import json
from pathlib import Path
from collections import Counter
from langdetect import detect, LangDetectException

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = ROOT / "datasets" / "feedback" / "feedback_data.json"

def load_feedback(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        print("Could not read feedback file or file empty.")
        return []
    if isinstance(data, dict):
        # maybe stored as { "feedback": [...] }
        for v in data.values():
            if isinstance(v, list):
                return v
        return []
    elif isinstance(data, list):
        return data
    return []

def main():
    if not FEEDBACK_PATH.exists():
        print("No feedback data found at", FEEDBACK_PATH)
        return
    fb = load_feedback(FEEDBACK_PATH)
    lang_counts = Counter()
    total = 0
    non_en = 0
    samples = []
    for r in fb:
        total += 1
        text = r.get("query") or r.get("text") or r.get("input") or ""
        try:
            lang = detect(text) if text else "unknown"
        except LangDetectException:
            lang = "unknown"
        lang_counts[lang] += 1
        if lang != "en":
            non_en += 1
            samples.append((lang, text))
    print(f"Total feedback records: {total}")
    print("Language counts (top):")
    for k, v in lang_counts.most_common():
        print(f"  {k}: {v}")
    print(f"Non-English feedback entries: {non_en}")
    if samples:
        print("\nExamples of non-English feedback (first 5):")
        for lang, txt in samples[:5]:
            print(f"  [{lang}] {txt}")

if __name__ == "__main__":
    main()