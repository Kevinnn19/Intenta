import re

def clean_text(text: str) -> str:
    """
    Clean and normalizer user input text.
    """
    text = text.lower()
    text = re.sub(r"\s+"," ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()
    return text

def clean_batch(texts: list[str]) -> list[str]:
    return [clean_text(text) for text in texts]