from app.retrieval.hybrid.hybrid_retriever import HybridRetriever

class RetrievalManager:
    VALID_DOMAINS = {
        "technical",
        "billing",
        "account"
    }

    def __init__(self):
        self.retriever = HybridRetriever()

    def retrieve(self, query, source, intent):
        if source not in self.VALID_DOMAINS:
            return []

        return self.retriever.retrieve(
            query=query,
            domain=source,
            intent=intent
        )