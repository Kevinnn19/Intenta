import pickle
from pathlib import Path
import faiss
import numpy as np
from app.inference.embedder import embedder

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

VECTORSTORE_DIR = BASE_DIR / "vectorstores"

class SemanticRetriever:
    def __init__(self):
        self.loaded_index = {}

    def load_domain(self, domain):
        if domain in self.loaded_index:
            return self.loaded_index[domain]

        domain_path = VECTORSTORE_DIR / domain

        idx_path = domain_path / "faiss_index.bin"
        metadata_path = domain_path / "metadata.pkl"

        idx = faiss.read_index(str(idx_path))

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        self.loaded_index[domain] = {
            "index": idx,
            "metadata": metadata,
        }

        return self.loaded_index[domain]

    def retrieve(self, query, domain, intent=None, top_k=5):
        domain_data = self.load_domain(domain)

        idx = domain_data["index"]
        metadata = domain_data["metadata"]

        query_embedding = embedder.embed_text(query)
        query_embedding = np.array([query_embedding]).astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = idx.search(query_embedding, top_k * 3)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            item = metadata[idx]

            if intent is not None:
                if item["intent"] != intent:
                    continue

            results.append({
                "text": item["text"],
                "intent": item["intent"],
                "score": round(float(score), 4)
            })

            if len(results) >= top_k:
                break

        return results