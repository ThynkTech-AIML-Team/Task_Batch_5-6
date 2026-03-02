import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors, Word2Vec, FastText
import gensim.downloader as api

print("Loading GloVe embeddings (small version for speed)...")
glove_vectors = api.load("glove-wiki-gigaword-50") 
print("GloVe loaded!")
def word_similarity(word1, word2, model):
    sim = model.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {sim:.3f}")

word_similarity('king', 'queen', glove_vectors)
word_similarity('man', 'woman', glove_vectors)
word_similarity('apple', 'orange', glove_vectors)

def word_analogy(a, b, c, model):
    result = model.most_similar(positive=[c, b], negative=[a], topn=1)
    print(f"{a} - {b} + {c} ≈ {result[0][0]} (score: {result[0][1]:.3f})")

word_analogy('man', 'king', 'woman', glove_vectors)  

words_to_plot = ['king', 'queen', 'man', 'woman', 'apple', 'orange', 'banana', 'fruit', 'doctor', 'nurse', 'hospital', 'medicine']
word_vecs = np.array([glove_vectors[w] for w in words_to_plot])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(word_vecs)

plt.figure(figsize=(8,6))
plt.scatter(pca_result[:,0], pca_result[:,1])

for i, word in enumerate(words_to_plot):
    plt.annotate(word, xy=(pca_result[i,0], pca_result[i,1]))

plt.title("PCA Visualization of Word Embeddings")
plt.show()
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
tsne_result = tsne.fit_transform(word_vecs)

plt.figure(figsize=(8,6))
plt.scatter(tsne_result[:,0], tsne_result[:,1])

for i, word in enumerate(words_to_plot):
    plt.annotate(word, xy=(tsne_result[i,0], tsne_result[i,1]))

plt.title("t-SNE Visualization of Word Embeddings")
plt.show()

print("\nLoading Word2Vec and FastText models (small versions)...")
word2vec_model = api.load("word2vec-google-news-300")  
fasttext_model = api.load("fasttext-wiki-news-subwords-300")  
print("Models loaded!")

print("\nAnalogy Test: man - king + woman")
print("GloVe: ", glove_vectors.most_similar(positive=['woman','king'], negative=['man'], topn=1))
print("Word2Vec: ", word2vec_model.most_similar(positive=['woman','king'], negative=['man'], topn=1))
print("FastText: ", fasttext_model.most_similar(positive=['woman','king'], negative=['man'], topn=1))

oov_word = "technobabble"
print("\nHandling out-of-vocabulary words with FastText:")
print("FastText vector for OOV word exists? ->", fasttext_model.has_index_for(oov_word))
print("Trying to get vector for OOV word using FastText...")
vec = fasttext_model[oov_word] 
print("Vector length:", len(vec))
