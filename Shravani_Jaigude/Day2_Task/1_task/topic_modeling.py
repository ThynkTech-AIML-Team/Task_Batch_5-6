import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def load_dataset(categories=None, n_samples=2000):
    data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        categories=categories
    )
    texts = data.data[:n_samples]
    return texts


def get_vectorizers(max_features=5000):
    count_vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2,
        max_features=max_features
    )

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2,
        max_features=max_features
    )

    return count_vectorizer, tfidf_vectorizer


def train_lda(dtm, n_topics=8):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch"
    )
    lda.fit(dtm)
    return lda


def train_nmf(tfidf, n_topics=8):
    nmf = NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=400
    )
    nmf.fit(tfidf)
    return nmf


def get_top_words(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_ids = topic.argsort()[:-n_top_words - 1:-1]
        words = [feature_names[i] for i in top_ids]
        topics.append((topic_idx, words))
    return topics


def coherence_score_from_topics(topics_words, tokenized_docs):
    """
    topics_words: list of list of words
    tokenized_docs: list of token lists
    """
    dictionary = Dictionary(tokenized_docs)
    coherence_model = CoherenceModel(
        topics=topics_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v"
    )
    return coherence_model.get_coherence()


def tokenize_docs(raw_docs):
    tokenized = []
    for doc in raw_docs:
        words = [w.lower() for w in doc.split() if w.isalpha()]
        if len(words) > 3:
            tokenized.append(words)
    return tokenized
