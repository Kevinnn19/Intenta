import json
from pathlib import Path
from typing import List
import numpy as np

from app.retrieval.semantic.semantic_retriever import SemanticRetriever
from app.retrieval.hybrid.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]

EVAL_FILE = ROOT / "datasets" / "retrieval_eval.json"

TOP_K = 5

semantic_retriever = SemanticRetriever()
hybrid_retriever = HybridRetriever()


def load_eval(path):
    if not path.exists():
        print("No retrieval eval file found:", path)
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def semantic_search(query, domain, top_k=TOP_K):
    results = semantic_retriever.retrieve(
        query=query,
        domain=domain,
        top_k=top_k
    )

    return [item["text"] for item in results]


def hybrid_search(query, domain, top_k=TOP_K):
    results = hybrid_retriever.retrieve(
        query=query,
        domain=domain,
        top_k=top_k
    )

    return [item["text"] for item in results]


def main():

    tests = load_eval(EVAL_FILE)

    if not tests:
        return

    semantic_correct = 0
    hybrid_correct = 0

    total = len(tests)

    for t in tests:

        query = t["query"]
        domain = t["domain"]
        relevant_texts = t["relevant_texts"]

        semantic_results = semantic_search(
            query=query,
            domain=domain,
            top_k=TOP_K
        )

        hybrid_results = hybrid_search(
            query=query,
            domain=domain,
            top_k=TOP_K
        )

        semantic_hit = False
        hybrid_hit = False

        for rel in relevant_texts:

            if any(rel.lower() in doc.lower() for doc in semantic_results):
                semantic_hit = True

            if any(rel.lower() in doc.lower() for doc in hybrid_results):
                hybrid_hit = True

        if semantic_hit:
            semantic_correct += 1

        if hybrid_hit:
            hybrid_correct += 1

    semantic_score = semantic_correct / total
    hybrid_score = hybrid_correct / total

    print("\n==============================")
    print("Retrieval Evaluation Results")
    print("==============================")

    print(f"Total Queries: {total}")

    print(
        f"Semantic Recall@{TOP_K}: "
        f"{semantic_score:.4f}"
    )

    print(
        f"Hybrid Recall@{TOP_K}: "
        f"{hybrid_score:.4f}"
    )

    if semantic_score > 0:

        improvement = (
            (hybrid_score - semantic_score)
            / semantic_score
        ) * 100

        print(
            f"Relative Improvement: "
            f"{improvement:.2f}%"
        )

    else:
        print(
            "Semantic score is 0. "
            "Cannot calculate improvement."
        )


if __name__ == "__main__":
    main()