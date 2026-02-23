import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# -----------------------------
# Build a semantic corpus
# -----------------------------

corpus = [
    "king queen prince royal crown palace",
    "man woman boy girl family",
    "dog cat animal pet",
    "car bus train vehicle transport",
    "python java programming coding software",
    "doctor nurse hospital medicine health",
    "football cricket sports match team",
    "money market finance trading stock",
] * 10   # repeat to strengthen signal

# -----------------------------
# Create word embeddings via TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

words = vectorizer.get_feature_names_out()
word_vectors = X.T.toarray()   # word-level vectors

word_to_index = {w: i for i, w in enumerate(words)}

# -----------------------------
# Similarity search
# -----------------------------

def nearest(word, top=5):
    if word not in word_to_index:
        return "Word not found"

    idx = word_to_index[word]
    sims = cosine_similarity([word_vectors[idx]], word_vectors)[0]
    order = sims.argsort()[::-1][1:top+1]
    return [(words[i], round(sims[i], 3)) for i in order]


print("\nSimilar to 'king':")
print(nearest("king"))

print("\nSimilar to 'doctor':")
print(nearest("doctor"))

# -----------------------------
# Analogy function
# king - man + woman ≈ queen
# -----------------------------

def analogy(a, b, c, top=5):
    for w in (a, b, c):
        if w not in word_to_index:
            return "Word missing in vocab"

    vec = (
        word_vectors[word_to_index[a]]
        - word_vectors[word_to_index[b]]
        + word_vectors[word_to_index[c]]
    )

    sims = cosine_similarity([vec], word_vectors)[0]
    order = sims.argsort()[::-1][:top]
    return [words[i] for i in order]


print("\nAnalogy: king - man + woman ≈")
print(analogy("king", "man", "woman"))

# -----------------------------
# Visualization with t-SNE
# -----------------------------

sample_words = [
    "king","queen","man","woman",
    "dog","cat",
    "car","bus",
    "doctor","nurse",
    "football","cricket"
]

sample_indices = [word_to_index[w] for w in sample_words]
vecs = word_vectors[sample_indices]

tsne = TSNE(
    n_components=2,
    perplexity=5,
    random_state=42,
    init="random"
)

coords = tsne.fit_transform(vecs)

plt.figure(figsize=(10, 8))

for i, w in enumerate(sample_words):
    x, y = coords[i]

    plt.scatter(x, y)

    # offset labels + arrows to avoid overlap
    plt.annotate(
        w,
        (x, y),
        xytext=(x + 1.0, y + 1.0),
        arrowprops=dict(arrowstyle="->", lw=0.6),
        fontsize=10
    )

plt.title("Word Embedding Explorer — t-SNE Projection")
plt.grid(True)
plt.tight_layout()
plt.show()