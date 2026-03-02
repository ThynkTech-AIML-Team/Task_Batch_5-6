import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==========================================
# 1. LOAD PRETRAINED MODELS [cite: 9, 11]
# ==========================================
print("Loading models... (This may take a few minutes usually 5-10 mins)")

# Load GloVe (using gensim's downloader for convenience)
# This corresponds to the 'glove.6B' dataset mentioned in the task
glove_model = api.load('glove-wiki-gigaword-100') 

# Load FastText (for the OOV bonus task)
# We need a model with subword information for OOV to work
fasttext_model = api.load('fasttext-wiki-news-subwords-300')

print("Models loaded successfully!")

# ==========================================
# 2. WORD SIMILARITY & ANALOGY TASKS 
# ==========================================
def perform_analogy(model, word1, word2, word3):
    """
    Solves analogies like: word1 - word2 + word3 = ?
    Example: King - Man + Woman = Queen
    """
    try:
        # positive=[word1, word3] means adding these vectors
        # negative=[word2] means subtracting this vector
        result = model.most_similar(positive=[word1, word3], negative=[word2], topn=1)
        print(f"{word1} - {word2} + {word3} = {result[0][0]} (Score: {result[0][1]:.2f})")
    except KeyError as e:
        print(f"Word not found in vocabulary: {e}")

print("\n--- 2. Word Analogies (GloVe) ---")
perform_analogy(glove_model, 'king', 'man', 'woman')      # Expected: Queen
perform_analogy(glove_model, 'paris', 'france', 'italy')  # Expected: Rome
perform_analogy(glove_model, 'computer', 'coder', 'human') # Experimental

# ==========================================
# 3. VISUALIZATION (PCA & t-SNE) 
# ==========================================
def visualize_embeddings(model, words):
    """
    Plots word embeddings in 2D space using PCA.
    """
    # Extract vectors for the chosen words
    vectors = []
    valid_words = []
    
    for word in words:
        if word in model:
            vectors.append(model[word])
            valid_words.append(word)
    
    if not vectors:
        print("No valid words found to visualize.")
        return

    # Reduce dimensions using PCA (Principal Component Analysis)
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='blue', edgecolors='k')

    for i, word in enumerate(valid_words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), xytext=(5, 2), 
                     textcoords='offset points', ha='right', va='bottom')
    
    plt.title("Word Embedding Clusters (PCA Visualization)")
    plt.grid(True)
    plt.show()

print("\n--- 3. Visualization ---")
# List of words to visualize clusters
word_list = [
    'king', 'queen', 'prince', 'princess',  # Royalty
    'apple', 'banana', 'orange', 'grape',   # Fruit
    'car', 'bus', 'train', 'bicycle',       # Vehicles
    'python', 'java', 'c++', 'code'         # Tech
]
visualize_embeddings(glove_model, word_list)

# ==========================================
# 4. FASTTEXT VS GLOVE (OOV HANDLING) 
# ==========================================
print("\n--- 4. Out-of-Vocabulary (OOV) Analysis ---")

# A misspelled word or complex compound word unlikely to be in GloVe
oov_word = "artificialintelligence" # or try 'gpus' or 'microservices' if older models

# Test on GloVe
if oov_word in glove_model:
    print(f"GloVe: Found '{oov_word}'!")
else:
    print(f"GloVe: Could not find '{oov_word}' (It treats words as atomic units).")

# Test on FastText
# FastText uses n-grams (sub-words), so it can construct a vector for unseen words
if oov_word in fasttext_model:
    print(f"FastText: Found '{oov_word}' directly.")
else:
    # Even if the exact word isn't in the vocab, FastText can often still generate a vector
    # Note: Gensim's implementation handles OOV lookup automatically for FastText if installed correctly
    try:
        vector = fasttext_model[oov_word]
        similar = fasttext_model.most_similar(oov_word, topn=3)
        print(f"FastText: Generated vector for '{oov_word}' using sub-words.")
        print(f"Most similar words to '{oov_word}': {similar}")
    except KeyError:
        print("FastText: Could not generate vector.")