import streamlit as st
import pandas as pd
import pyLDAvis
from pyLDAvis.lda_model import prepare


from topic_modeling import (
    load_dataset,
    get_vectorizers,
    train_lda,
    train_nmf,
    get_top_words,
    tokenize_docs,
    coherence_score_from_topics
)

st.set_page_config(page_title="Topic Modeling Dashboard", layout="wide")

st.title("Topic Modeling Research Dashboard")
st.write("Dataset: **20 Newsgroups** | Models: **LDA + NMF**")

# Sidebar
st.sidebar.header(" Settings")
n_samples = st.sidebar.slider("Number of Documents", 500, 4000, 2000, 500)
n_topics = st.sidebar.slider("Number of Topics", 2, 15, 8)
n_top_words = st.sidebar.slider("Top Words per Topic", 5, 20, 10)
max_features = st.sidebar.slider("Max Features", 2000, 15000, 5000, 1000)

st.sidebar.info("Tip: Use 2000 samples for fast results.")

# Load dataset
docs = load_dataset(n_samples=n_samples)

count_vectorizer, tfidf_vectorizer = get_vectorizers(max_features=max_features)

dtm = count_vectorizer.fit_transform(docs)
tfidf = tfidf_vectorizer.fit_transform(docs)

# Train models
lda_model = train_lda(dtm, n_topics=n_topics)
nmf_model = train_nmf(tfidf, n_topics=n_topics)

# Feature names
count_features = count_vectorizer.get_feature_names_out()
tfidf_features = tfidf_vectorizer.get_feature_names_out()

# Topics
lda_topics = get_top_words(lda_model, count_features, n_top_words=n_top_words)
nmf_topics = get_top_words(nmf_model, tfidf_features, n_top_words=n_top_words)

# Tokenize for coherence
tokenized_docs = tokenize_docs(docs)

lda_topic_words = [words for _, words in lda_topics]
nmf_topic_words = [words for _, words in nmf_topics]

lda_coh = coherence_score_from_topics(lda_topic_words, tokenized_docs)
nmf_coh = coherence_score_from_topics(nmf_topic_words, tokenized_docs)

lda_perplexity = lda_model.perplexity(dtm)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader(" LDA Topics")
    for idx, words in lda_topics:
        st.write(f"**Topic {idx+1}:** " + ", ".join(words))

with col2:
    st.subheader(" NMF Topics")
    for idx, words in nmf_topics:
        st.write(f"**Topic {idx+1}:** " + ", ".join(words))

st.divider()

st.subheader(" Model Comparison")
df_compare = pd.DataFrame({
    "Model": ["LDA", "NMF"],
    "Coherence (c_v)": [lda_coh, nmf_coh],
    "Perplexity": [lda_perplexity, None]
})
st.dataframe(df_compare, use_container_width=True)

st.divider()

st.subheader(" pyLDAvis Visualization (LDA)")

vis = prepare(lda_model, dtm, count_vectorizer)
html_string = pyLDAvis.prepared_data_to_html(vis)

st.components.v1.html(html_string, height=800, scrolling=True)
