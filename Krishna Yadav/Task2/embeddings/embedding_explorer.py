print("Word Embedding Explorer Started...")

import gensim.downloader as api
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import numpy as np

# -------------------------------------------------------
# Load Pretrained Embeddings
# -------------------------------------------------------
print("Loading GloVe embeddings...")
glove = api.load("glove-wiki-gigaword-100")

print("Loading Word2Vec embeddings...")
word2vec = api.load("word2vec-google-news-300")

print("Loading FastText embeddings...")
fasttext = api.load("fasttext-wiki-news-subwords-300")

# -------------------------------------------------------
# Word Similarity Example
# -------------------------------------------------------
print("\nSimilarity Example (GloVe):")
print(glove.most_similar("king"))

# -------------------------------------------------------
# Analogy Example
# king - man + woman ≈ queen
# -------------------------------------------------------
print("\nAnalogy Example (GloVe):")
print(glove.most_similar(positive=["king", "woman"], negative=["man"]))

# -------------------------------------------------------
# FastText OOV Example
# -------------------------------------------------------
print("\nFastText OOV Example:")
print("Vector exists for 'hellooo'? ", "hellooo" in fasttext.key_to_index)

# -------------------------------------------------------
# t-SNE Visualization
# -------------------------------------------------------
print("\nPreparing t-SNE Visualization...")

words = random.sample(list(glove.key_to_index.keys()), 200)

vectors = [glove[word] for word in words]

# -------------------------------------------------------
# FIX: convert list -> numpy array so TSNE works
# -------------------------------------------------------
vectors = np.array(vectors)

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

X_tsne = tsne.fit_transform(vectors)

plt.figure(figsize=(10, 8))

for i, word in enumerate(words):
    x, y = X_tsne[i]
    plt.scatter(x, y)
    plt.text(x + 0.3, y + 0.3, word, fontsize=8)

plt.title("GloVe Word Embedding t-SNE Visualization")

plt.show()
