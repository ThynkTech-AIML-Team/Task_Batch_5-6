print("News Similarity Application Started...")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from topic_modeling.dataset_loader import load_data
import numpy as np

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
print("Loading news dataset...")
docs = load_data()

# Use subset for faster execution
documents = docs[:500]

print("Total documents loaded:", len(documents))

# -------------------------------------------------------
# Load Embedding Model
# -------------------------------------------------------
print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------------
# Encode Documents
# -------------------------------------------------------
print("Encoding documents into embeddings...")
doc_embeddings = model.encode(documents)

# -------------------------------------------------------
# Function: Find Similar Articles
# -------------------------------------------------------
def find_similar_articles(query, top_k=3):

    print("\nEncoding query...")
    query_embedding = model.encode([query])

    print("Calculating cosine similarity...")
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Sort by similarity
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append((similarities[idx], documents[idx]))

    return results

# -------------------------------------------------------
# Example Query
# -------------------------------------------------------
if __name__ == "__main__":

    query_text = "space shuttle launch mission nasa orbit"

    print("\nQuery:", query_text)

    results = find_similar_articles(query_text)

    print("\nTop Similar Articles:\n")

    for score, text in results:
        print("Similarity Score:", score)
        print(text[:300])
        print("-" * 80)
