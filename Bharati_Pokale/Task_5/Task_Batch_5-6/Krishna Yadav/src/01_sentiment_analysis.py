print(">>> Script started")

import nltk
import random
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -----------------------------
# Load Dataset
# -----------------------------
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

texts = [" ".join(words) for words, category in documents]
labels = [category for words, category in documents]

print("Total documents:", len(texts))


# -----------------------------
# Bag of Words Model
# -----------------------------
vectorizer = CountVectorizer(max_features=3000)
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("BoW Accuracy:", accuracy_score(y_test, y_pred))


# -----------------------------
# TF-IDF Model
# -----------------------------
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_tfidf, labels, test_size=0.2, random_state=42
)

model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train_t, y_train_t)

y_pred_t = model_tfidf.predict(X_test_t)
print("TF-IDF Accuracy:", accuracy_score(y_test_t, y_pred_t))
print(">>> Script finished")
