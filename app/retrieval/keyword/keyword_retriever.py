import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

VECTORSTORE_DIR = BASE_DIR / "vectorstores"

class KeywordRetriever:

    def __init__(self):
        self.loaded_domains = {}
        print("Keyword retriever initialized successfully.")

    def load_domain(self, domain):
        if domain in self.loaded_domains:
            return self.loaded_domains[domain]

        metadata_path = (
            VECTORSTORE_DIR / domain / "metadata.pkl"
        )

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        documents = [
            item["text"]
            for item in metadata
        ]

        tokenized_docs = [
            doc.lower().split() for doc in documents
        ]

        bm25 = BM25Okapi(tokenized_docs)

        self.loaded_domains[domain] = {
            "bm25": bm25,
            "metadata": metadata
        }

        print(f"{domain} keyword index loaded.")

        return self.loaded_domains[domain]

    def retrieve(self, query, domain, intent=None, top_k=5):
        domain_data = self.load_domain(domain)

        bm25 = domain_data["bm25"]
        metadata = domain_data["metadata"]

        tokenized_query = query.lower().split()

        scores = bm25.get_scores(tokenized_query)
        scored_docs = list(zip(metadata, scores))

        scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        results = []

        for item, score in scored_docs:
            if intent is not None:
                if item["intent"] != intent:
                    continue

            results.append({
                "text": item["text"],
                "intent": item["intent"],
                "score": round(float(score),4)
            })

            if len(results) >= top_k:
                break

        return results