# Task 3 – NLP Internship Assignment (Final)

## 📌 Overview

This project implements **Task 3 – NLP Internship Assignment** with a focus on text classification, word embeddings, and a mini NLP application. It demonstrates end-to-end NLP workflow including preprocessing, feature extraction, model building, evaluation, and interpretation.

All work is implemented using Python in Jupyter Notebooks.

---

## ✅ Part 1 — Basic Text Classification Project

### Objective

Build and compare multiple machine learning models for text classification.

### Dataset Used

Text classification dataset (Spam/Sentiment dataset as used in the notebook).

### Preprocessing Steps

* Converted text to lowercase
* Removed punctuation and special characters
* Removed stopwords
* Cleaned and normalized text

### Feature Extraction

* CountVectorizer (Bag of Words)
* TF-IDF Vectorizer

### Models Implemented

* Naive Bayes
* Logistic Regression
* Support Vector Machine (where applicable)

### Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

### Comparison

* Compared CountVectorizer vs TF-IDF results
* Compared model performance
* Extracted top important words contributing to each class

---

## ✅ Part 2 — Word Embedding Mini Task

### Objective

Understand and test pretrained word embeddings.

### Tasks Completed

* Loaded pretrained Word2Vec / GloVe embeddings
* Found most similar words
* Performed analogy operations (example: king − man + woman)
* Reduced dimensions using PCA
* Visualized selected words in 2D space

### Observations

* Similar words cluster together
* Vector arithmetic captures semantic relationships
* PCA plots show meaningful grouping

---

## ✅ Part 3 — Mini NLP Application — **Option A: Fake News Detection System**

### Objective

Build a machine learning model to classify news articles as **Fake** or **Real**.

### Method Used

* Text preprocessing and cleaning
* TF-IDF vectorization
* Model training using Logistic Regression / Naive Bayes

### Outputs Generated

* Prediction of Fake vs Real news
* Accuracy score
* Confusion Matrix visualization
* Classification Report

### Feature Interpretation

* Displayed top important words contributing to Fake and Real classes
* Showed which terms most influenced predictions

---

## 🛠 Tech Stack

* Python
* Jupyter Notebook
* scikit-learn
* pandas / numpy
* matplotlib
* nltk / gensim (as required)

---

## 📂 Folder Structure

* `/Codes` → All notebooks and scripts
* `/Screenshots` → Output screenshots
* `/README.md` → Project documentation

---

## 📊 Results Summary

* Built multiple NLP classification models
* Compared vectorization techniques
* Demonstrated embedding behavior
* Successfully implemented a Fake News Detection system with evaluation metrics and feature insights

---

## ✅ Submission Includes

* Clean Python notebooks
* Model training & evaluation outputs
* Visualizations
* Screenshots of results
* This README file

---
