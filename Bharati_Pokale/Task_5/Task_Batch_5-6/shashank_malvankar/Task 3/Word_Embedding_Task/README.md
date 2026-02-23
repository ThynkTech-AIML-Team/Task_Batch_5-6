# Word Embedding Mini Task – Pretrained GloVe Model

## Overview
This project demonstrates the use of pretrained word embeddings to understand semantic relationships between words. Word embeddings represent words as numerical vectors that capture meaning, similarity, and relationships in language.

Pretrained GloVe embeddings were used to perform similarity analysis, analogy reasoning, and visualization.

---

## Objective
The objective of this task was to:

- Load pretrained word embeddings
- Find similar words using vector similarity
- Perform analogy operations using vector arithmetic
- Visualize word relationships using PCA
- Compare similarity scores between words

---

## Model Used

**Model:** GloVe (Global Vectors for Word Representation)

**Source:** Wikipedia + Gigaword corpus  
**Embedding Size:** 50 dimensions  
**Library:** gensim

GloVe is a pretrained word embedding model that learns vector representations of words based on their co-occurrence statistics.

---

## Tasks Performed

### 1. Load Pretrained Model
The pretrained GloVe model was loaded using gensim downloader.

### 2. Similar Word Detection
Used cosine similarity to find words with similar meanings.

Example:

king -> queen, prince, monarch

This shows that semantically related words have similar vector representations.

---

### 3. Analogy Task
Performed vector arithmetic to find relationships between words.

Example:

king - man + woman = queen


This demonstrates how embeddings capture semantic relationships.

---

### 4. Word Embedding Visualization
Selected multiple words and visualized their vector representations using PCA (Principal Component Analysis).

This reduced the embeddings from 50 dimensions to 2 dimensions for visualization.

The plot showed that:

- Similar words cluster together
- Opposite or unrelated words are farther apart

---

### 5. Similarity Comparison
Compared similarity scores between different word pairs.

Example:

| Word Pair | Similarity |
|---------|------------|
| king - queen | High |
| king - computer | Lower |
| good - bad | Moderate |

This confirms embeddings capture semantic similarity.

---

## Results and Observations

- Word embeddings successfully captured semantic meaning
- Related words had higher similarity scores
- Analogy tasks produced correct results
- Visualization showed meaningful clustering of related words
- Pretrained embeddings provide powerful semantic understanding

---

## Files Included

- `word_embedding.ipynb` : Jupyter Notebook containing implementation
- `outputs/` : Screenshots of results and visualizations
- `README.md` : Documentation

---

## Conclusion

This task demonstrated the effectiveness of pretrained word embeddings in representing semantic relationships between words. GloVe embeddings successfully enabled similarity detection, analogy reasoning, and visualization of word relationships.

Word embeddings are widely used in NLP applications such as sentiment analysis, chatbots, recommendation systems, and search engines.

