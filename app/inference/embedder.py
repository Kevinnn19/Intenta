from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self):
        """
        Load embedding model once during startup.
        """
        self.model_name = "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert single text into embedding vector.
        """
        embedding = self.model.encode(
            text, convert_to_numpy=True
        )
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Comvert multiple texts into embeddings.
        """
        embeddings = self.model.encode(
            texts, convert_to_numpy=True
        )
        return embeddings

embedder = Embedder()