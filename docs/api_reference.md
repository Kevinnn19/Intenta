# Intenta API Reference

Base URL:

```text
http://localhost:8000
```

---

# GET /

Returns API status.

### Response

```json
{
  "message": "Intenta API is running"
}
```

---

# GET /health

Health check endpoint.

### Response

```json
{
  "status": "healthy"
}
```

---

# POST /predict

Predicts user intent and retrieves relevant documents.

### Request

```json
{
  "text": "my model training failed"
}
```

### Response

```json
{
  "prediction": "training_error",
  "confidence": 0.87,
  "language": "en",
  "source": "technical",
  "retrieved_docs": []
}
```

---

# POST /feedback

Stores user feedback.

### Request

```json
{
  "query": "my password is not working",
  "predicted_intent": "login_issue",
  "source": "account",
  "feedback": "positive"
}
```

### Response

```json
{
  "message": "Feedback recorded successfully"
}
```

---

# GET /analytics

Returns feedback analytics.

### Response

```json
{
  "total_feedback": 4,
  "positive_feedback": 3,
  "negative_feedback": 1,
  "positive_rate": 75,
  "negative_rate": 25,
  "most_failed_intent": "training_error",
  "most_successful_intent": "login_issue"
}
```

---

## Swagger Documentation

Available at:

```text
http://localhost:8000/docs
```

---

## Response Fields

### Prediction Response

| Field          | Description         |
| -------------- | ------------------- |
| prediction     | Predicted intent    |
| confidence     | Model confidence    |
| language       | Detected language   |
| source         | Routed domain       |
| retrieved_docs | Retrieved documents |
| alternatives   | Alternative intents |
| latency_ms     | Request latency     |

---

## Supported Domains

```text
technical
billing
account
```

---

## Supported Languages

```text
English
Hindi
Gujarati
```
