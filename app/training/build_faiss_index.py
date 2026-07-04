import json
import pickle
import numpy as np
from pathlib import Path
import faiss
from app.inference.embedder import embedder

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_PATH = BASE_DIR / "datasets" / "domains"

VECTORSTORE_DIR = BASE_DIR / "vectorstores"

DOMAINS = [
    "technical",
    "billing",
    "account"
]

for domain in DOMAINS:

    dataset_path = (DATASET_PATH / f"{domain}_data.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) == 0:
        continue

    texts = [item["text"] for item in data]

    embeddings = embedder.embed_batch(texts)
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    domain_vector_dir = VECTORSTORE_DIR / domain
    domain_vector_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(domain_vector_dir / "faiss_index.bin"))

    with open(
        domain_vector_dir / "metadata.pkl", "wb"
    ) as f:
        pickle.dump(data, f)