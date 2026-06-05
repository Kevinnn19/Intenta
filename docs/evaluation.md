# Intenta Evaluation Report

## Intent Classification Evaluation

### Model

```text
TF-IDF + Logistic Regression
```

### Dataset

Custom Developer Intent Dataset

Number of Intents:

```text
11
```

---

## Classification Results

| Metric          | Value  |
| --------------- | ------ |
| Accuracy        | 72.09% |
| Macro Precision | 0.70   |
| Macro Recall    | 0.69   |
| Macro F1 Score  | 0.67   |

---

## Per-Intent Performance

| Intent              | F1 Score |
| ------------------- | -------- |
| git_issue           | 1.00     |
| payment_failed      | 1.00     |
| project_transfer    | 1.00     |
| database_connection | 0.86     |
| model_saving        | 0.86     |
| docker_issue        | 0.75     |
| training_error      | 0.75     |
| deployment_issue    | 0.57     |
| dependency_issue    | 0.60     |
| login_issue         | 0.00     |
| project_run         | 0.00     |

---

## Hybrid Retrieval Evaluation

### Semantic Retrieval

Model:

```text
all-MiniLM-L6-v2
```

Vector Database:

```text
FAISS
```

---

### Keyword Retrieval

Algorithm:

```text
BM25
```

---

### Hybrid Search

Formula:

```text
0.7 × Semantic Score
+
0.3 × Keyword Score
```

---

## Example Retrieval Result

Query:

```text
save my trained model
```

Top Result:

```text
Use joblib.dump(model, 'model.pkl') to save trained sklearn models
```

Intent:

```text
model_saving
```

---

## Multilingual Evaluation

### English

Input:

```text
my password is not working
```

Detected:

```text
en
```

Prediction:

```text
login_issue
```

---

### Hindi

Input:

```text
मेरा पासवर्ड काम नहीं कर रहा
```

Detected:

```text
hi
```

Prediction:

```text
login_issue
```

---

### Gujarati

Input:

```text
મારું પાસવર્ડ કામ કરતું નથી
```

Detected:

```text
gu
```

Prediction:

```text
login_issue
```

---

## Conclusion

The system successfully combines:

* Intent Classification
* Domain Routing
* Hybrid Retrieval
* Multilingual Query Processing
* Feedback Analytics

into a unified support-query understanding platform.
