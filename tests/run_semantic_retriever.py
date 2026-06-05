from app.retrieval.semantic.semantic_retriever import SemanticRetriever

retriever = SemanticRetriever()

query = "how do i save my trained model"

results = retriever.retrieve(query)

for r in results:
    print(r)