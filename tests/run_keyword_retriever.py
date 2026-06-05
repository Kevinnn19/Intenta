from app.retrieval.keyword.keyword_retriever import KeywordRetriever

retriever = KeywordRetriever()

results = retriever.retrieve(
    query = "uvicorn app.main:app --reload",
    domain = "technical"
)

for result in results:
    print(result)