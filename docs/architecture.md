# Intenta Architecture

## System Overview

Intenta is a multilingual intent classification and hybrid retrieval system designed to understand user support queries and retrieve relevant domain-specific knowledge.

The system combines machine learning, semantic search, keyword retrieval, language processing, and REST APIs.

---

## High-Level Architecture

```text
User Query
    ↓
Language Detection
    ↓
Translation (if required)
    ↓
Text Cleaning
    ↓
Intent Classification
(TF-IDF + Logistic Regression)
    ↓
Intent Routing
    ↓
Retrieval Manager
    ↓
Hybrid Retrieval
├── Semantic Search (FAISS)
└── BM25 Keyword Search
    ↓
Score Fusion
    ↓
Retrieved Documents
    ↓
API Response
```

---

## Components

### Language Processing Layer

Responsible for:

* Language detection
* Query translation
* Query normalization

Supported languages:

* English
* Hindi
* Gujarati

Files:

```text
app/language/
├── language_detector.py
├── translator.py
└── language_service.py
```

---

### Intent Classification Layer

Responsible for predicting the user intent.

Model:

* TF-IDF Vectorizer
* Logistic Regression

Files:

```text
app/training/train.py
app/inference/predictor.py
```

Output:

```json
{
  "prediction": "login_issue",
  "confidence": 0.87
}
```

---

### Routing Layer

Maps predicted intents to knowledge domains.

Domains:

* technical
* billing
* account

Files:

```text
app/routing/
├── routing_config.py
└── intent_router.py
```

---

### Retrieval Layer

Uses hybrid search.

Components:

#### Semantic Search

Embedding Model:

```text
all-MiniLM-L6-v2
```

Vector Database:

```text
FAISS
```

Files:

```text
app/retrieval/semantic/
```

#### Keyword Search

Algorithm:

```text
BM25
```

Files:

```text
app/retrieval/keyword/
```

#### Hybrid Search

Score Formula:

```text
Final Score =
0.7 × Semantic Score
+
0.3 × Keyword Score
```

Files:

```text
app/retrieval/hybrid/
```

---

### Feedback Layer

Stores user feedback.

Feedback Types:

* positive
* negative

Files:

```text
app/feedback/
```

---

### Analytics Layer

Generates:

* Positive feedback rate
* Negative feedback rate
* Most successful intent
* Most failed intent

Files:

```text
app/analytics/
```

---

## Storage Architecture

### Models

```text
models/
├── developer_intent_classifier.joblib
└── intent_classifier.joblib
```

### Vector Stores

```text
vectorstores/
├── technical/
├── billing/
└── account/
```

### Feedback Storage

```text
datasets/feedback/feedback_data.json
```

---

## API Layer

Built using FastAPI.

Files:

```text
app/main.py
```

Available APIs:

* /predict
* /feedback
* /analytics
* /health

---

## Future Architecture Improvements

* Docker Deployment
* PostgreSQL Storage
* MLflow Tracking
* CI/CD Pipelines
* Cloud Deployment
* Automated Retraining
