from app.inference.embedder import embedder
from sklearn.metrics.pairwise import cosine_similarity

text1 = "how to save model"
text2 = "how can i store trained model"
text3 = "weather today"

emb1 = embedder.embed_text(text1)
emb2 = embedder.embed_text(text2)
emb3 = embedder.embed_text(text3)

score1 = cosine_similarity([emb1], [emb2])[0][0]
score2 = cosine_similarity([emb1], [emb3])[0][0]

print("Semantic Match: ", score1)
print("Unrelated Match: ", score2)