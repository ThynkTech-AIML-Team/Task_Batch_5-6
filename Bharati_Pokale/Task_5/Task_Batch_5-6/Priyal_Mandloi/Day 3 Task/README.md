\# Day 3 – NLP Internship Assignment



This repository contains solutions for the Day 3 NLP Internship tasks, covering text classification, word embeddings, and a mini NLP application.



---



\## Task 1: Basic Text Classification



\### Dataset

\- SMS Spam Collection Dataset

\- File: `spam.csv`

\- Labels: spam / ham



\### Preprocessing

\- Converted text to lowercase

\- Removed punctuation

\- Removed stopwords



\### Feature Extraction

\- CountVectorizer

\- TF-IDF Vectorizer



\### Models Used

\- Naive Bayes

\- Logistic Regression



\### Evaluation Metrics

\- Accuracy

\- Confusion Matrix

\- Classification Report



\### Results

\- Naive Bayes achieved the best accuracy (~96.7%)

\- Logistic Regression achieved ~95% accuracy

\- Naive Bayes performed better for spam detection



---



\## Task 2: Word Embeddings



\### Models Used

\- Pretrained Word2Vec (Google News)



\### Tasks Performed

\- Found similar words (e.g., king → queen)

\- Performed analogy task (king − man + woman = queen)

\- Visualized word embeddings using PCA



\### Observation

Word embeddings successfully captured semantic relationships between words.



---



\## Task 3: Mini NLP Application – Rule-Based FAQ Chatbot



\### Description

A simple rule-based FAQ chatbot was built using NLP techniques.



\### Techniques Used

\- TF-IDF Vectorization

\- Cosine Similarity



\### Functionality

\- Matches user queries with predefined questions

\- Returns the most relevant answer



---



\## Files in This Project

\- `Day3\_Task1\_Basic\_Text\_Classification.ipynb`

\- `Day3\_Task2\_Word\_Embeddings.ipynb`

\- `Day3\_Task3\_Mini\_NLP\_Application.ipynb`

\- `spam.csv`

\- `README.md`



