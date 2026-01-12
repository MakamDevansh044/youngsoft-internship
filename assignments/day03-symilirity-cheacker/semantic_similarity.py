from sentence_transformers import SentenceTransformer
import numpy as np
from itertools import combinations

# -------------------------------
# Step 1: List of sentences
# -------------------------------
sentences = [
    "I love machine learning",
    "Machine learning is amazing",
    "I enjoy playing cricket",
    "Cricket is my favorite sport"
]

# -------------------------------
# Step 2: Load a free embedding model
# -------------------------------
# 'all-mpnet-base-v2' is widely used & high quality
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Compute embeddings
embeddings = model.encode(sentences)

# -------------------------------
# Step 3: Cosine similarity function
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------------
# Step 4: Compare sentence pairs
# -------------------------------
similarity_scores = []
top_pair = None
max_similarity = -1

for i, j in combinations(range(len(sentences)), 2):
    sim = cosine_similarity(embeddings[i], embeddings[j])
    similarity_scores.append((sentences[i], sentences[j], sim))
    
    if sim > max_similarity:
        max_similarity = sim
        top_pair = (sentences[i], sentences[j], sim)

# -------------------------------
# Step 5: Output
# -------------------------------
print("\nSimilarity Scores:\n")
for s1, s2, score in similarity_scores:
    print(f"'{s1}' <-> '{s2}' : {score:.4f}")

print("\nTop Most Similar Sentences:")
print(f"'{top_pair[0]}' <-> '{top_pair[1]}' : {top_pair[2]:.4f}")
