# Intenta - Intelligent Intent Classification & Retrieval System

## Overview

Intenta is an end-to-end Machine Learning system that classifies user intents and retrieves relevant domain-specific knowledge using a hybrid search architecture.

The project combines traditional NLP, semantic embeddings, vector search, and REST APIs to provide intelligent support-query understanding.

---

## Example Workflow

```text
User Query
    ↓
Language Detection
    ↓
Translation (if required)
    ↓
Text Preprocessing
    ↓
Intent Classification
(TF-IDF + Logistic Regression)
    ↓
Intent Routing
    ↓
Hybrid Retrieval
(FAISS + BM25)
    ↓
Relevant Response Documents
```

---

## Features

### Intent Classification

* TF-IDF Vectorization
* Logistic Regression Classifier
* Confidence-based Prediction
* Alternative Intent Suggestions

### Hybrid Retrieval

* Semantic Search using FAISS
* Keyword Search using BM25
* Hybrid Ranking Strategy
* Domain-Specific Knowledge Bases

### Multilingual Support

Supported Languages:

* English
* Hindi
* Gujarati

Queries are automatically translated to English before classification.

### Analytics & Feedback

* User Feedback Collection
* Positive/Negative Feedback Tracking
* Intent Performance Analytics
* Failure Trend Monitoring

### API Services

* Prediction API
* Semantic Search API
* Feedback API
* Analytics API
* Health Check API

---

## Architecture

```text
                    ┌─────────────────┐
                    │ User Query      │
                    └────────┬────────┘
                             │
                             ▼
                 ┌─────────────────────┐
                 │ Language Detection  │
                 └────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │ Translation Layer   │
                 └────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │ Text Cleaning       │
                 └────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │ Intent Classifier   │
                 │ TF-IDF + LogReg     │
                 └────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │ Intent Router       │
                 └────────┬────────────┘
                          │
                          ▼
          ┌──────────────────────────────────┐
          │ Hybrid Retrieval System          │
          │                                  │
          │ Semantic Search (FAISS)          │
          │ Keyword Search (BM25)            │
          └──────────────┬───────────────────┘
                         │
                         ▼
                 ┌─────────────────────┐
                 │ Retrieved Documents │
                 └─────────────────────┘
```

---

## Tech Stack

| Category           | Technology            |
| ------------------ | --------------------- |
| Backend            | FastAPI               |
| ML Framework       | Scikit-Learn          |
| Embeddings         | Sentence Transformers |
| Vector Search      | FAISS                 |
| Keyword Search     | BM25                  |
| Language Detection | LangDetect            |
| Translation        | Deep Translator       |
| Model Storage      | Joblib                |
| Data Storage       | JSON                  |
| Evaluation         | Scikit-Learn Metrics  |
| Visualization      | Matplotlib, Seaborn   |

---

## Machine Learning Pipeline

### Classification Model

```text
TF-IDF Vectorizer
        +
Logistic Regression
```

### Retrieval Model

```text
SentenceTransformer
(all-MiniLM-L6-v2)
        +
FAISS
```

### Hybrid Scoring

```text
Final Score =
(0.7 × Semantic Score)
+
(0.3 × Keyword Score)
```

---

## Dataset

### Intent Dataset

* 11 Intent Classes
* Developer Support Queries
* Training Examples per Intent

Example Intents:

```text
api_error
database_connection
deployment_issue
docker_issue
git_issue
login_issue
model_saving
payment_failed
project_run
project_transfer
training_error
```

### Domain Knowledge Bases

```text
technical
billing
account
```

---

## API Endpoints

| Method | Endpoint           | Description                           |
| ------ | ------------------ | ------------------------------------- |
| POST   | `/predict`         | Predict intent and retrieve documents |
| POST   | `/semantic/search` | Semantic search endpoint              |
| POST   | `/feedback`        | Submit user feedback                  |
| GET    | `/analytics`       | View feedback analytics               |
| GET    | `/health`          | Health check                          |

---

## Predict Intent Example

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
  "source": "technical"
}
```

---

## Results

| Metric              | Value   |
| ------------------- | ------- |
| Accuracy            | 72.09%  |
| Intent Classes      | 11      |
| Languages Supported | 3       |
| Retrieval Method    | Hybrid  |
| API Framework       | FastAPI |

---

## Running Locally

### Clone Repository

```bash
git clone <repository-url>
cd Intenta
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python app/training/train.py
```

### Build FAISS Indexes

```bash
python app/training/build_faiss_index.py
```

### Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

### Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

## Project Highlights

* End-to-End ML Pipeline
* Intent Classification System
* Hybrid Search Architecture
* Multilingual Query Processing
* FastAPI REST API
* Feedback Analytics
* FAISS Vector Search
* BM25 Retrieval
* Modular Architecture

---

## Future Improvements

* Docker Deployment
* CI/CD Pipeline
* MLflow Integration
* PostgreSQL Storage
* Automated Retraining
* Monitoring & Observability
* Model Registry
* Cloud Deployment

---

## Author

**Kevin**

Machine Learning Engineer | MLOps Enthusiast
