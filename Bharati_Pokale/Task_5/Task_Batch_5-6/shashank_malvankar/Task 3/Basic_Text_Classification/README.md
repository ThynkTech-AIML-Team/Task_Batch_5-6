# Basic Text Classification Project – IMDb Sentiment Analysis

## Overview
This project implements a text classification system to classify IMDb movie reviews as **positive** or **negative**. Multiple machine learning models were trained and evaluated using different text vectorization techniques.

---

## Dataset
- **Name:** IMDb Large Movie Review Dataset
- **Source:** Stanford AI Lab
- **Size:** 50,000 movie reviews
- **Classes:**
  - Positive
  - Negative
- **Distribution:** Balanced (25,000 positive, 25,000 negative)

Each review is stored as raw text and labeled with its sentiment.

---

## Text Preprocessing
The following preprocessing steps were applied:

- Converted text to lowercase
- Removed punctuation
- Removed English stopwords
- Converted cleaned text into numerical vectors using:
  - CountVectorizer
  - TF-IDF Vectorizer

---

## Models Used

The following machine learning models were trained and evaluated:

1. Naive Bayes (MultinomialNB)
2. Logistic Regression
3. Support Vector Machine (LinearSVC)

Each model was trained using CountVectorizer and TF-IDF representations.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

---

## Results

| Model | Vectorizer | Accuracy |
|------|------------|----------|
| Naive Bayes | CountVectorizer | ~86% |
| Logistic Regression | CountVectorizer | ~89% |
| SVM | CountVectorizer | ~91% |
| Logistic Regression | TF-IDF | ~92–93% |

TF-IDF with Logistic Regression achieved the best performance.

---

## Key Observations

- TF-IDF performed better than CountVectorizer
- Logistic Regression and SVM outperformed Naive Bayes
- Important words identified matched expected sentiment patterns
- The model successfully classified positive and negative reviews with high accuracy

---

## Files Included

- `text_classification.ipynb` – Main Jupyter Notebook
- `dataset/imdb.csv` – Processed dataset
- `outputs/` – Screenshots and results
- `README.md` – Project documentation

---

## Conclusion

This project demonstrates a complete NLP pipeline including:

- Text preprocessing
- Feature extraction
- Model training
- Model evaluation
- Result interpretation

The best performing model achieved over **92% accuracy** on the IMDb sentiment classification task.
