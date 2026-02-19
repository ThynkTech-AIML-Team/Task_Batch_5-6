import gensim.downloader as api
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt  # type: ignore
import os
import numpy as np
from typing import Any

def run_embedding_explorer():
    print("📥 Loading GloVe embeddings (this takes a moment)...")
    # 'Any' prevents the editor from thinking 'model' is just a string
    model: Any = api.load("glove-wiki-gigaword-50")
    
    # 1. Analogy Task: king - man + woman ≈ queen
    print("\n🕵️ Solving Analogy: king - man + woman...")
    result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
    print(f"Result: {result[0][0]} (Confidence: {result[0][1]:.4f})")

    # 2. t-SNE Visualization logic
    words_to_plot = ['king', 'queen', 'man', 'woman', 'apple', 'orange', 'fruit']
    word_vectors = np.array([model[w] for w in words_to_plot])

    print("📊 Generating t-SNE plot...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=3, init='pca', learning_rate='auto')
    vectors_2d = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(8, 6))
    for i, word in enumerate(words_to_plot):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
        plt.annotate(word, (vectors_2d[i, 0] + 0.1, vectors_2d[i, 1] + 0.1))
    
    plt.title("Word Embedding Clusters (t-SNE)")
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/embedding_plot.png")
    
    # 3. Generate Research Report
    report_path = "outputs/embedding_research_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== WORD EMBEDDING RESEARCH REPORT ===\n")
        f.write(f"Analogy Result: king - man + woman = {result[0][0]}\n")
        f.write("\nFastText vs GloVe:\n")
        f.write("FastText handles Out-of-Vocabulary (OOV) words better because it uses sub-word info.")

    print(f"✅ Success! Report and Plot saved in 'outputs/'")

if __name__ == "__main__":
    run_embedding_explorer()