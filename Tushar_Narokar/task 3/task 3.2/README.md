#  Word Embedding Mini Task

> **Objective:** Understand basic word embeddings — similarity, analogies, and 2D visualization using pretrained Word2Vec and GloVe models.

---

##  Project Structure

```
task 3.2/
 venv/                          # Python 3.11.1 virtual environment
 word_embedding.ipynb           # Main Jupyter Notebook (13 cells)
 requirements.txt               # Pinned dependencies
 pca_word2vec.png               # PCA plot — Word2Vec
 pca_glove.png                  # PCA plot — GloVe
 comparison_bar_chart.png       # Bar chart — W2V vs GloVe similarity
 pca_comparison_side_by_side.png# Side-by-side PCA comparison
```

---

##  Dataset

No custom dataset is used. Both models are **pretrained on large public corpora** and loaded directly via [`gensim.downloader`](https://radimrehurek.com/gensim/downloader.html).

| Model | Training Corpus | Size |
|-------|----------------|------|
| Word2Vec | Google News (approx. 100 billion words) | ~1.7 GB |
| GloVe | Wikipedia 2014 + Gigaword 5 (6 billion tokens) | ~128 MB |

Both are downloaded automatically on first run and **cached locally** — subsequent runs load instantly from disk.

---

##  Models Used

### 1. Word2Vec — `word2vec-google-news-300`
- **Algorithm:** Skip-gram / CBOW (predictive, neural)
- **Dimensions:** 300
- **Vocabulary:** ~3 million words
- **Strength:** Excellent at capturing local context; strong on analogies
- **Loaded via:** `gensim.downloader.load('word2vec-google-news-300')`

### 2. GloVe — `glove-wiki-gigaword-100`
- **Algorithm:** Global Vectors (count-based, co-occurrence matrix + log-bilinear)
- **Dimensions:** 100
- **Vocabulary:** ~400,000 words
- **Strength:** Captures global statistical patterns across the corpus
- **Loaded via:** `gensim.downloader.load('glove-wiki-gigaword-100')`

---

##  Tasks & Results

###  1. Similar Words (Top-5 by Cosine Similarity)

| Query | Word2Vec Results | GloVe Results |
|-------|-----------------|---------------|
| `king` | queen, prince, monarch, kings, royal | queen, prince, royal, throne, kings |
| `computer` | computers, laptop, software, PC, desktop | computers, software, electronic, digital, hardware |
| `music` | musical, jazz, rock, pop, songs | musical, song, songs, band, genre |
| `france` | germany, spain, italy, paris, europe | french, paris, germany, european, europe |
| `doctor` | physician, nurse, patient, surgeon, medical | physician, nurse, medical, hospital, patient |

> Word2Vec tends to return higher cosine similarity scores (0.7–0.85) vs GloVe (0.65–0.80) due to richer 300-dim vector space.

---

###  2. Word Analogies

Formula: **`result = positive_words − negative_words`**

| Analogy | Word2Vec Answer | GloVe Answer |
|---------|----------------|--------------|
| king − man + woman | **queen**  | **queen**  |
| paris − france + germany | **berlin**  | **berlin**  |
| doctor − man + woman | **nurse** / gynecologist | **nurse**  |
| biggest − big + small | **smallest**  | **smallest**  |

> Both models successfully resolve the classic `king − man + woman = queen` analogy, demonstrating that word embeddings encode semantic relationships geometrically.

---

###  3. PCA Visualization (2D)

Words from 5 semantic categories were reduced from high-dimensional space to 2D using **Principal Component Analysis (PCA)**:

| Category | Words |
|----------|-------|
|  Royalty | king, queen, prince, princess |
|  Countries | france, germany, italy, spain |
|  Technology | computer, software, internet |
|  Professions | doctor, teacher, engineer |
|  Music | music, guitar, piano |

**Key Observation:** Words from the same semantic group **cluster together** in 2D space, confirming that both models learn meaningful representations. Countries and royalty terms are particularly well-separated from technology/music terms.

---

###  Bonus: Word2Vec vs GloVe Comparison

| Feature | Word2Vec | GloVe |
|---------|----------|-------|
| Avg. similarity score (top-5) | Higher (~0.75) | Moderate (~0.70) |
| Analogy accuracy | Excellent | Good |
| Vocabulary coverage | Larger (~3M) | Smaller (~400K) |
| Vector dimensions | 300 | 100 |
| Training paradigm | Predictive (local context) | Count-based (global statistics) |

> **Conclusion:** Word2Vec produces sharper similarity scores and stronger analogies due to its larger corpus and higher-dimensional vectors. GloVe, while slightly lower in similarity scores, is faster to load and still captures semantic structure effectively.

---

##  How to Run

```powershell
# 1. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 2. Launch Jupyter Notebook
jupyter notebook word_embedding.ipynb
```

Then select kernel **`Python 3.11.1 (Word Embedding)`** and run all cells top-to-bottom.

>  **First run only:** Cells 2 & 3 download the pretrained models (~1.8 GB total). Ensure an active internet connection.

---

##  Environment

- **Python:** 3.11.1
- **Key Libraries:** `gensim 4.4.0`, `numpy 2.4.2`, `matplotlib 3.10.8`, `scikit-learn 1.8.0`
- **Notebook:** JupyterLab 4.5.4 / Notebook 7.5.3
