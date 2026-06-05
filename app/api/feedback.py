from pydantic import BaseModel
from typing import Literal

class FeedbackRequest(BaseModel):
    query: str
    predicted_intent: str
    source: str
    feedback: Literal["positive", "negative"]
