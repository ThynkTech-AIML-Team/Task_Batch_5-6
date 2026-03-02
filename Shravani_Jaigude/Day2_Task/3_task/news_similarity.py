import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_news_data(n_samples=2000):
    data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes")
    )
    docs = data.data[:n_samples]
    return docs


def build_vectorizer(docs):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    X = vectorizer.fit_transform(docs)
    return vectorizer, X


def get_top_k_similar(input_text, vectorizer, X, docs, k=3):
    query_vec = vectorizer.transform([input_text])
    sims = cosine_similarity(query_vec, X).flatten()

    top_indices = sims.argsort()[-k:][::-1]
    results = []

    for idx in top_indices:
        results.append({
            "similarity": float(sims[idx]),
            "text": docs[idx]
        })

    return results
