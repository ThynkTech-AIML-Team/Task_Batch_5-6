import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pyLDAvis
import pyLDAvis

# ----------------------------
# Download NLTK (first run)
# ----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_documents(documents):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        tokens = text.split()
        tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stop_words and len(word) > 3
        ]
        return " ".join(tokens)

    return [preprocess(doc) for doc in documents]

# ----------------------------
# Load Dataset (Cached)
# ----------------------------
@st.cache_data
def load_data():
    dataset = fetch_20newsgroups(remove=('headers','footers','quotes'))
    return dataset.data

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📊 Topic Modeling Research Dashboard")

documents = load_data()
processed_docs = preprocess_documents(documents)

model_choice = st.selectbox("Select Model", ["LDA", "NMF"])
num_topics = st.slider("Select Number of Topics", 5, 15, 10)

texts = [doc.split() for doc in processed_docs]
dictionary = Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.9)

# ----------------------------
# LDA MODEL
# ----------------------------
if model_choice == "LDA":

    vectorizer = CountVectorizer(max_df=0.9, min_df=5)
    doc_term_matrix = vectorizer.fit_transform(processed_docs)

    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42
    )
    lda_model.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()

    lda_topics = [
        [feature_names[i] for i in topic.argsort()[:-11:-1]]
        for topic in lda_model.components_
    ]

    coherence_model = CoherenceModel(
        topics=lda_topics,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()
    perplexity = lda_model.perplexity(doc_term_matrix)

    st.subheader("LDA Results")
    st.write("Coherence Score:", coherence_score)
    st.write("Perplexity:", perplexity)

    st.subheader("Topics")
    for idx, topic in enumerate(lda_topics):
        st.write(f"Topic {idx}: {topic}")

# ----------------------------
# NMF MODEL
# ----------------------------
elif model_choice == "NMF":

    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=5)
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

    nmf_model = NMF(
        n_components=num_topics,
        random_state=42
    )
    nmf_model.fit(tfidf_matrix)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    nmf_topics = [
        [tfidf_feature_names[i] for i in topic.argsort()[:-11:-1]]
        for topic in nmf_model.components_
    ]

    coherence_model = CoherenceModel(
        topics=nmf_topics,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()

    st.subheader("NMF Results")
    st.write("Coherence Score:", coherence_score)

    st.subheader("Topics")
    for idx, topic in enumerate(nmf_topics):
        st.write(f"Topic {idx}: {topic}")
