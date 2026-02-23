

import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



print("Loading GloVe...")
glove = api.load("glove-wiki-gigaword-100")

print("Loading FastText...")
fasttext = api.load("fasttext-wiki-news-subwords-300")

print("Models loaded successfully!\n")



print("Word Similarity (king vs queen):")
print("GloVe:", glove.similarity("king", "queen"))
print("FastText:", fasttext.similarity("king", "queen"))



print("\nWord Analogy (king - man + woman):")

print("GloVe:")
print(glove.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
))

print("\nFastText:")
print(fasttext.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
))


print("\nOOV Handling Demonstration:")

oov_word = "datascientist"

try:
    glove[oov_word]
    print("GloVe: Word found (unexpected)")
except KeyError:
    print("GloVe: OOV ❌")

print(
    "FastText: Uses subword (character n-gram) information to handle\n"
    "out-of-vocabulary and rare words. Due to Gensim KeyedVectors\n"
    "limitations, subword vector generation is explained conceptually."
)



words = [
    "king", "queen", "man", "woman",
    "car", "bike", "bus", "train",
    "computer", "laptop", "keyboard", "mouse"
]

vectors = np.array([glove[word] for word in words])

pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(reduced[i, 0], reduced[i, 1])
    plt.text(reduced[i, 0] + 0.01, reduced[i, 1] + 0.01, word)

plt.title("PCA Visualization of Word Embeddings (GloVe)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()
