from app.retrieval.semantic.semantic_retriever import SemanticRetriever
from app.retrieval.keyword.keyword_retriever import KeywordRetriever

class HybridRetriever:
    def __init__(self):
        self.semantic = SemanticRetriever()
        self.keyword = KeywordRetriever()
        print("Hybrid retriever initialized successfully.")

    def retrieve(self, query, domain, intent=None, top_k=5):
        semantic_results = self.semantic.retrieve(
            query=query,
            domain=domain,
            intent=intent,
            top_k=top_k
        )

        keyword_results = self.keyword.retrieve(
            query=query,
            domain=domain,
            intent=intent,
            top_k=top_k
        )

        if not semantic_results and not keyword_results:
            return []

        max_keyword = max(
            [item["score"] for item in keyword_results],
            default=1
        )
        # Avoid division by zero when all keyword scores are 0
        if max_keyword == 0:
            max_keyword = 1

        combined = {}

        for item in semantic_results:
            key = item["text"]

            combined[key] = {
                "text": item["text"],
                "intent": item["intent"],
                "semantic_score":item["score"],
                "keyword_score": 0
            }

        for item in keyword_results:
            key = item["text"]

            if key not in combined:
                combined[key] = {
                    "text": item["text"],
                    "intent": item["intent"],
                    "semantic_score": 0,
                    "keyword_score": item["score"] / max_keyword
                }
            else:
                combined[key]["keyword_score"] = (
                    item["score"] / max_keyword
                )

        results = []

        for item in combined.values():

            final_score = (
                item["semantic_score"] * 0.7 + item["keyword_score"] * 0.3
            )

            if final_score > 0:
                results.append({
                    "text": item["text"],
                    "intent": item["intent"],
                    "score": round(final_score, 4),
                    "semantic_score": round(item["semantic_score"], 4),
                    "keyword_score": round(item["keyword_score"], 4)
                })

        results.sort(
            key=lambda x: x["score"], reverse=True
        )

        return results[:top_k]