import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from gensim.models import Word2Vec, FastText

print("STEP 1: Loading GloVe embeddings...")

embeddings = {}

with open(r"C:\Users\DELL\Downloads\Aditya_Wagh\Day-2\Task_2\glove.6B\glove.6B.300d.txt", encoding="utf8") as f:
    
    for line in f:
        
        values = line.split()
        
        word = values[0]
        
        vector = np.array(values[1:], dtype="float32")
        
        embeddings[word] = vector

print("Total GloVe words loaded:", len(embeddings))


# STEP 2
print("\nSTEP 2: Word similarity")

similarity = cosine_similarity(
    [embeddings["king"]],
    [embeddings["queen"]]
)

print("Similarity between king and queen:", similarity[0][0])


# STEP 3

print("\nSTEP 3: Word analogy")

result = embeddings["king"] - embeddings["man"] + embeddings["woman"]

similarity = cosine_similarity(
    [result],
    [embeddings["queen"]]
)

print("king - man + woman similar to queen:", similarity[0][0])



# STEP 4

print("\nSTEP 4: PCA visualization")

words = ["king", "queen", "man", "woman", "apple", "orange"]

vectors = [embeddings[word] for word in words]

pca = PCA(n_components=2)

result = pca.fit_transform(vectors)

plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(words):
    
    plt.annotate(word, (result[i, 0], result[i, 1]))

plt.title("GloVe Word Embeddings Visualization")

plt.show()


# STEP 5

print("\nSTEP 5: Word2Vec vs FastText")

sentences = [
    ["king", "queen", "man", "woman"],
    ["apple", "orange", "fruit"],
    ["car", "bike", "vehicle"]
]


w2v = Word2Vec(sentences, vector_size=100, min_count=1)


fast = FastText(sentences, vector_size=100, min_count=1)

print("Word2Vec similarity king-queen:",
      w2v.wv.similarity("king", "queen"))

print("FastText similarity king-queen:",
      fast.wv.similarity("king", "queen"))


# STEP 6

print("\nSTEP 6: FastText unknown word test")

try:
    
    print("Word2Vec unknown word:", 
          w2v.wv["kingdom"])
    
except:
    
    print("Word2Vec cannot handle unknown word")


print("FastText unknown word vector exists:",
      fast.wv["kingdom"][:5])
