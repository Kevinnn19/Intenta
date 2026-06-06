from fastapi import FastAPI

from app.api.feedback import FeedbackRequest
from app.feedback.feedback_service import feedback_service
from app.inference.predictor import predict_intent
from pydantic import BaseModel
from app.analytics.feedback_analytics import analyze_feedback

app = FastAPI(
    title="Intenta AI API",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Intenta API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: PredictionRequest):
    text = data.text
    result = predict_intent(text)
    return result

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    feedback_service.save_feedback(
        query=request.query,
        predicted_intent=request.predicted_intent,
        source=request.source,
        feedback=request.feedback
    )
    return {
        "message": "Feedback recorded successfully"
    }

@app.get("/analytics")
def analytics():
    return analyze_feedback()