
import os, zipfile, numpy as np, requests, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec, FastText


def download_glove():
    if not os.path.exists("glove.6B.50d.txt"):
        print("Downloading GloVe (82 MB) ...")
        r = requests.get("https://nlp.stanford.edu/data/glove.6B.zip", stream=True)
        with open("glove.6B.zip", "wb") as f:
            for chunk in r.iter_content(32768): f.write(chunk)
        with zipfile.ZipFile("glove.6B.zip") as z:
            z.extract("glove.6B.50d.txt")
        print("Done.")

def load_glove(max_words=20000):
    download_glove()
    print(f"Loading GloVe (top {max_words} words)...")
    vecs = {}
    with open("glove.6B.50d.txt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_words: break
            parts = line.split()
            vecs[parts[0]] = np.array(parts[1:], dtype=np.float32)
    print(f"Loaded {len(vecs)} vectors.")
    return vecs


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def similar(word, vecs, n=5):
    if word not in vecs:
        return f"'{word}' not in vocabulary"
    scores = sorted(vecs.items(), key=lambda x: cosine(vecs[word], x[1]), reverse=True)
    return [(w, round(float(cosine(vecs[word], v)), 4)) for w, v in scores[1:n+1]]

def analogy(a, b, c, vecs, n=3):
    if any(w not in vecs for w in [a, b, c]):
        return "One or more words not in vocabulary"
    target = vecs[b] - vecs[a] + vecs[c]
    exclude = {a, b, c}
    scores = sorted(
        [(w, cosine(target, v)) for w, v in vecs.items() if w not in exclude],
        key=lambda x: x[1], reverse=True
    )
    return [(w, round(float(s), 4)) for w, s in scores[:n]]


WORD_GROUPS = {
    "royalty":   ["king", "queen", "prince", "princess"],
    "places":    ["paris", "london", "berlin", "rome"],
    "tech":      ["computer", "network", "language", "learning"],
    "nature":    ["river", "mountain", "ocean", "forest"],
}
COLORS = ["#E05C5C", "#5C9BE0", "#5CBF7A", "#E0A95C"]

def visualize(vecs, title="Embeddings", method="pca", save_as=None):
    words, groups, matrix = [], [], []
    for i, (group, group_words) in enumerate(WORD_GROUPS.items()):
        for w in group_words:
            if w in vecs:
                words.append(w)
                groups.append(i)
                matrix.append(np.array(vecs[w]))

    if not words:
        print("No words found for visualization.")
        return

    matrix = np.array(matrix)
    if method == "tsne":
        coords = TSNE(n_components=2, perplexity=min(5, len(words)-1),
                      random_state=42, max_iter=500).fit_transform(matrix)
    else:
        coords = PCA(n_components=2).fit_transform(matrix)

    plt.figure(figsize=(8, 6))
    for i, (word, group_idx) in enumerate(zip(words, groups)):
        x, y = coords[i]
        plt.scatter(x, y, color=COLORS[group_idx % len(COLORS)], s=80, zorder=3)
        plt.annotate(word, (x, y), fontsize=9, xytext=(5, 4), textcoords="offset points")

    legend = [mpatches.Patch(color=COLORS[i], label=g) for i, g in enumerate(WORD_GROUPS)]
    plt.legend(handles=legend, fontsize=8)
    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()

    fname = save_as or f"{method}.png"
    plt.savefig(fname, dpi=150)
    print(f"  Saved → {fname}")
    plt.close()


CORPUS = [
    "the king ruled the kingdom".split(),
    "the queen ruled alongside the king".split(),
    "the man worked in the field".split(),
    "the woman ran the household".split(),
    "paris is the capital of france".split(),
    "berlin is the capital of germany".split(),
    "london is the capital of england".split(),
    "machine learning uses neural networks".split(),
    "natural language processing understands text".split(),
    "the river flows through the mountain valley".split(),
    "forests oceans and rivers are natural habitats".split(),
    "the prince and princess lived in the palace".split(),
]

def train_models():
    print("Training Word2Vec...", end=" ", flush=True)
    w2v = Word2Vec(CORPUS, vector_size=50, window=3, min_count=1, epochs=200, sg=1)
    print("done")

    print("Training FastText...", end=" ", flush=True)
    ft = FastText(CORPUS, vector_size=50, window=3, min_count=1, epochs=200, min_n=2, max_n=4)
    print("done")

    return w2v, ft


def oov_demo(glove, w2v, ft):
    test_words = ["king", "running", "footballish", "covid", "machinelearning"]
    print(f"\n  {'Word':<20} {'GloVe':^10} {'Word2Vec':^10} {'FastText':^10}")
    print("  " + "─" * 50)
    for word in test_words:
        g  = "✓" if word in glove else "✗ OOV"
        w  = "✓" if word in w2v.wv else "✗ OOV"
        ft_status = "✓" if word in ft.wv else "≈ subword"
        print(f"  {word:<20} {g:^10} {w:^10} {ft_status:^10}")
    print("\n  ≈ subword = FastText synthesises a vector from character n-grams")


def main():

    print("\n=== 1. Loading GloVe ===")
    glove = load_glove()

    print("\n=== 2. Word Similarity (GloVe) ===")
    for word in ["king", "paris", "ocean"]:
        print(f"\n  Similar to '{word}':")
        for w, s in similar(word, glove):
            print(f"    {w:<18} {s}")

    print("\n=== 3. Analogy Tasks  [b - a + c ≈ ?] ===")
    for a, b, c in [("man", "king", "woman"), ("paris", "france", "berlin")]:
        print(f"\n  {b} - {a} + {c}  ≈  {analogy(a, b, c, glove)}")

    print("\n=== 4. Visualization ===")
    visualize(glove, title="GloVe — PCA",  method="pca",  save_as="glove_pca.png")
    visualize(glove, title="GloVe — tSNE", method="tsne", save_as="glove_tsne.png")

    print("\n=== 5. Training Word2Vec & FastText ===")
    w2v, ft = train_models()

    w2v_vecs = {w: w2v.wv[w] for w in w2v.wv.index_to_key}
    ft_vecs  = {w: ft.wv[w]  for w in ft.wv.index_to_key}

    visualize(w2v_vecs, title="Word2Vec — PCA", method="pca", save_as="w2v_pca.png")
    visualize(ft_vecs,  title="FastText — PCA", method="pca", save_as="ft_pca.png")

    print("\n=== 6. OOV: GloVe vs Word2Vec vs FastText ===")
    oov_demo(glove, w2v, ft)

    print("\nDone! Check the .png files for visualizations.")

if __name__ == "__main__":
    main()