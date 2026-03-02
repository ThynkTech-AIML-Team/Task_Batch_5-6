import streamlit as st
import pandas as pd
import pyLDAvis
from pyLDAvis import lda_model
import plotly.express as px
from sklearn.datasets import fetch_20newsgroups

from scripts.preprocessing import preprocess_text
from scripts.train_models import vectorize_docs, train_lda, train_nmf
from scripts.evaluation import lda_perplexity, coherence_score, display_topics

st.set_page_config(layout="wide")
st.title("Interactive Topic Modeling Dashboard: LDA vs NMF")


# Load & preprocess data
newsgroups = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
documents = preprocess_text(newsgroups.data)


# Sidebar controls
num_topics = st.sidebar.slider("Number of Topics", 5, 20, 10)

#Vectorization

doc_term_matrix, tfidf_matrix, count_vectorizer, tfidf_vectorizer = vectorize_docs(documents)


#Model Training
lda_model_obj = train_lda(doc_term_matrix, num_topics)
nmf_model_obj = train_nmf(tfidf_matrix, num_topics)


#Model Performance Metrics
st.header("Model Performance")
metrics = pd.DataFrame({
    "Model": ["LDA", "NMF"],
    "Coherence": [
        coherence_score(lda_model_obj, count_vectorizer.get_feature_names_out()),
        coherence_score(nmf_model_obj, tfidf_vectorizer.get_feature_names_out())
    ],
    "Perplexity": [lda_perplexity(lda_model_obj, doc_term_matrix), "N/A"]
})
st.table(metrics)

# LDA pyLDAvis Visualization
st.header("LDA Topic Visualization")
lda_vis = lda_model.prepare(
    lda_model_obj, 
    doc_term_matrix, 
    count_vectorizer, 
    mds="tsne"
)
st.components.v1.html(
    pyLDAvis.prepared_data_to_html(lda_vis),
    height=700,
    scrolling=True
)

# NMF Bubble Chart Visualization
st.header("NMF Topic Visualization (Bubble Chart)")

def prepare_nmf_bubbles(model, feature_names, top_n=10):
    data = {"Topic": [], "Word": [], "Weight": []}
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-top_n:][::-1]
        for i in top_indices:
            data["Topic"].append(f"Topic {topic_idx+1}")
            data["Word"].append(feature_names[i])
            data["Weight"].append(topic[i])
    return pd.DataFrame(data)

nmf_df = prepare_nmf_bubbles(nmf_model_obj, tfidf_vectorizer.get_feature_names_out())

fig = px.scatter(
    nmf_df, x="Topic", y="Weight", size="Weight", color="Topic",
    hover_data=["Word"], title="NMF Topics Bubble Chart"
)
st.plotly_chart(fig, use_container_width=True)


# Display top words side-by-side
st.header("Top Words in Each Topic")
col1, col2 = st.columns(2)

with col1:
    st.subheader("LDA Topics")
    lda_topics = display_topics(lda_model_obj, count_vectorizer.get_feature_names_out())
    for topic, words in lda_topics.items():
        st.write(f"**{topic}**: {words}")

with col2:
    st.subheader("NMF Topics")
    nmf_topics = display_topics(nmf_model_obj, tfidf_vectorizer.get_feature_names_out())
    for topic, words in nmf_topics.items():
        st.write(f"**{topic}**: {words}")
