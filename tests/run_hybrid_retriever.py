from app.retrieval.hybrid.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()

results = retriever.retrieve(
    query = "how do i save my trained model",
    domain = "technical",
    intent = "model_saving"
)

for result in results:
    print(result)