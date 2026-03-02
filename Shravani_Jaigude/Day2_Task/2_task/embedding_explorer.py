import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

GLOVE_PATH = "glove.6B.100d.txt"


def load_glove(path, limit=50000):
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def most_similar(word, embeddings, topn=5):
    if word not in embeddings:
        return []

    target_vec = embeddings[word]
    sims = []

    for w, vec in embeddings.items():
        if w == word:
            continue
        sim = cosine_similarity(target_vec, vec)
        sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:topn]


def analogy(a, b, c, embeddings, topn=5):
    """
    a - b + c = ?
    Example: king - man + woman = queen
    """
    for w in [a, b, c]:
        if w not in embeddings:
            return []

    vec = embeddings[a] - embeddings[b] + embeddings[c]

    sims = []
    for w, v in embeddings.items():
        if w in [a, b, c]:
            continue
        sim = cosine_similarity(vec, v)
        sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:topn]


def pca_plot(words, embeddings):
    vectors = []
    valid_words = []

    for w in words:
        if w in embeddings:
            valid_words.append(w)
            vectors.append(embeddings[w])

    vectors = np.array(vectors)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1])

    for i, word in enumerate(valid_words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]))

    plt.title("PCA Visualization of Word Embeddings (GloVe)")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(GLOVE_PATH):
        print("❌ GloVe file not found!")
        print("Download glove.6B.100d.txt and place it in this folder.")
        exit()

    print("Loading GloVe embeddings...")
    embeddings = load_glove(GLOVE_PATH, limit=50000)
    print("Loaded words:", len(embeddings))

    # Similarity
    word = "computer"
    print(f"\nMost similar to '{word}':")
    for w, s in most_similar(word, embeddings):
        print(w, "->", round(s, 4))

    # Analogy
    print("\nAnalogy: king - man + woman = ?")
    for w, s in analogy("king", "man", "woman", embeddings):
        print(w, "->", round(s, 4))

    # PCA Visualization
    sample_words = [
        "king", "queen", "man", "woman",
        "paris", "france", "london", "england",
        "computer", "software", "hardware", "internet",
        "doctor", "nurse", "teacher", "student"
    ]

    pca_plot(sample_words, embeddings)
