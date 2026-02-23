# Day 3 – NLP Internship Assignment

**Author:** Ayush Singh
**Internship:** NLP Internship
**Day:** 3

---

# Overview

This project implements three Natural Language Processing (NLP) tasks:

* Task 1: Text Classification (Spam Detection)
* Task 2: Word Embeddings Exploration
* Task 3: FAQ Chatbot Application

All tasks were implemented using Python, Scikit-learn, and Gensim in Jupyter Notebook.

---

# Task 1 – Text Classification (Spam Detection)

## Objective

Build a machine learning model to classify SMS messages as Spam or Ham.

## Steps Performed

* Loaded SMS Spam dataset
* Cleaned text (lowercase, punctuation removal, stopwords removal)
* Feature extraction using:

  * CountVectorizer
  * TF-IDF Vectorizer
* Trained models:

  * Naive Bayes
  * Logistic Regression
  * Support Vector Machine
* Evaluated models using:

  * Accuracy score
  * Confusion Matrix
  * Classification Report
* Extracted important spam words (Bonus)

## Result

Logistic Regression achieved highest accuracy (~98%).

## File

```
task1_text_classification.ipynb
```

---

# Task 2 – Word Embeddings

## Objective

Understand semantic relationships between words using pretrained embeddings.

## Steps Performed

* Loaded pretrained GloVe model
* Loaded Word2Vec model (Bonus)
* Found similar words
* Performed analogy task (king – man + woman = queen)
* Visualized word vectors using PCA
* Compared GloVe vs Word2Vec similarity (Bonus)
* Interactive similarity exploration

## File

```
task2_word_embeddings.ipynb
```

---

# Task 3 – FAQ Chatbot

## Objective

Build an interactive chatbot using TF-IDF and cosine similarity.

## Features

* Rule-based FAQ chatbot
* Uses TF-IDF vectorization
* Matches user question with stored FAQs
* Returns most relevant answer
* Interactive conversation support

## Example

```
User: what is python
Bot: Python is a programming language
```

## File

```
task3_faq_chatbot.ipynb
```

---

# Technologies Used

* Python
* Scikit-learn
* Gensim
* NLTK
* NumPy
* Matplotlib
* Jupyter Notebook

---

# Folder Structure

```
Day-3/
│
├── task1_text_classification.ipynb
├── task2_word_embeddings.ipynb
├── task3_faq_chatbot.ipynb
├── spam.csv
├── README.md
└── venv/
```

---

# Conclusion

Successfully implemented NLP applications including:

* Spam detection using machine learning
* Word embeddings analysis using pretrained models
* Interactive FAQ chatbot using NLP techniques

This demonstrates practical understanding of text preprocessing, feature extraction, word embeddings, and NLP applications.

---
