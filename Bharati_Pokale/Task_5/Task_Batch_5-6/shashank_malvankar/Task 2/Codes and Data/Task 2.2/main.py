from utils.load_glove import load_glove
from utils.similarity import similarity_glove, similarity_gensim
from utils.analogy import analogy_gensim

from models.word2vec_model import train_word2vec
from models.fasttext_model import train_fasttext

from visualization.pca_visualization import plot_pca
from visualization.tsne_visualization import plot_tsne

if __name__ == "__main__":

    print("Loading GloVe...")
    glove = load_glove("data/glove.6B.100d.txt")

    print("\nTraining Word2Vec...")
    w2v = train_word2vec()

    print("\nTraining FastText...")
    ft = train_fasttext()


    print("\nSimilarity Comparison:")

    print("GloVe:", similarity_glove("king", "queen", glove))
    print("Word2Vec:", similarity_gensim("king", "queen", w2v))
    print("FastText:", similarity_gensim("king", "queen", ft))


    print("\nAnalogy Comparison:")

    print("Word2Vec:",
          analogy_gensim("man", "king", "woman", w2v))

    print("FastText:",
          analogy_gensim("man", "king", "woman", ft))


    print("\nOOV Test (word not in vocab):")

    print("Word2Vec:", similarity_gensim("hellooo", "king", w2v))
    print("FastText:", similarity_gensim("hellooo", "king", ft))


    words = [
        "king",
        "queen",
        "man",
        "woman",
        "paris",
        "france",
        "london",
        "england"
    ]

    print("\nPCA Visualization...")
    plot_pca(words, glove)

    print("\nt-SNE Visualization...")
    plot_tsne(words, glove)
