# ThynkTech NLP Internship Tasks

## Task 1: Sentiment Analysis

This project demonstrates a basic NLP sentiment analysis system using the NLTK Movie Reviews dataset.

### Techniques Used
- Bag of Words (CountVectorizer)
- TF-IDF (TfidfVectorizer)
- Logistic Regression

### Workflow
1. Load movie review dataset from NLTK
2. Convert text into numerical features
3. Train sentiment classifier
4. Evaluate model performance

### Results
- BoW Accuracy: ~0.81
- TF-IDF Accuracy: ~0.81

### How to Run

```bash
conda activate nlp_env
python src/01_sentiment_analysis.py
