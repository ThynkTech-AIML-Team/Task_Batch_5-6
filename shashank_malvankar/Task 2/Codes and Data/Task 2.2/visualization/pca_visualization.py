from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_pca(words, embeddings):

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

    pca = PCA(n_components=2)

    result = pca.fit_transform(vectors)

    plt.figure()

    for i, word in enumerate(valid_words):

        plt.scatter(result[i, 0], result[i, 1])

        plt.text(
            result[i, 0],
            result[i, 1],
            word
        )

    plt.title("PCA Visualization")

    plt.show()
