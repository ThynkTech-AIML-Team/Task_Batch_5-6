from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne(words, embeddings):

    vectors = [
        embeddings[word]
        for word in words
        if word in embeddings
    ]

    valid_words = [
        word
        for word in words
        if word in embeddings
    ]

    # Convert list → numpy array (FIX)
    vectors = np.array(vectors)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=5
    )

    result = tsne.fit_transform(vectors)

    plt.figure()

    for i, word in enumerate(valid_words):

        plt.scatter(result[i, 0], result[i, 1])

        plt.text(
            result[i, 0],
            result[i, 1],
            word
        )

    plt.title("t-SNE Visualization")

    plt.show()
