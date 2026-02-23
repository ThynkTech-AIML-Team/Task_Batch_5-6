# NLP Internship Assignment - Day 3

## 📝 Overview
This project encompasses three core Natural Language Processing (NLP) tasks completed as part of an internship assignment. It covers the end-to-end pipeline of text processing, from classification and semantic analysis to entity extraction.

---
 
## 🚀 Tasks and Implementation

### 1. Basic Text Classification
**Objective:** Develop a robust system to classify SMS messages as "Ham" (legitimate) or "Spam".
- **Algorithms:** Naive Bayes, Logistic Regression, and Support Vector Machine (SVM).
- **Techniques:** 
    - Text Cleaning (Lowercase conversion, punctuation removal).
    - Stopword removal using NLTK.
    - Comparison between **CountVectorizer** and **TF-IDF Vectorization**.
- **Results:** 
    - Achieved an accuracy range of **97-98%**.
    - TF-IDF showed slightly superior performance in distinguishing spam features.
    - Top indicated spam words: *free, call, txt, mobile, stop*.

### 2. Word Embedding Mini Task
**Objective:** Analyze semantic relationships between words using high-dimensional vector representations.
- **Models:** 
    - **GloVe** (glove-wiki-gigaword-50)
    - **Word2Vec** (word2vec-google-news-300)
- **Key Features:**
    - **Similarity Analysis:** Identifying words with highest cosine similarity.
    - **Analogies:** Successful execution of the classic `king - man + woman = queen` analogy.
    - **Visualization:** 2D projection of word clusters using **PCA (Principal Component Analysis)**.

### 3. Mini NLP Application (NER Visualizer)
**Objective:** Create a visual tool to extract and categorize entities from raw text.
- **Framework:** spaCy.
- **Functionality:** 
    - Extracts crucial entities: `PERSON`, `ORG` (Organization), `GPE` (Geopolitical Entity), and `DATE`.
    - Provides a color-coded **displacy** visualization.
    - Outputs an entity frequency counter for quick text summarization.

---
## 👤 Author
- **Name:** Atharva Hanumant Admile
- **Batch:** 06
