import streamlit as st
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from preprocess import preprocess_documents
from dataset_loader import load_data

import pyLDAvis
import pyLDAvis.lda_model
import streamlit.components.v1 as components

# -------------------------------------------------------
# Title
# -------------------------------------------------------
st.title("📊 Topic Modeling Research Dashboard")

# -------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["NMF", "LDA"]
)

num_topics = st.sidebar.slider(
    "Number of Topics",
    2,
    10,
    5
)

# -------------------------------------------------------
# CACHE DATA LOADING (FAST)
# -------------------------------------------------------
@st.cache_data
def get_data():
    docs = load_data()
    return docs

# -------------------------------------------------------
# CACHE PREPROCESSING (VERY IMPORTANT)
# -------------------------------------------------------
@st.cache_data
def get_processed_docs(docs):
    processed_docs = preprocess_documents(docs[:500])
    return processed_docs

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------
st.write("Loading dataset...")
docs = get_data()

# -------------------------------------------------------
# Preprocess Text
# -------------------------------------------------------
st.write("Preprocessing text...")
processed_docs = get_processed_docs(docs)

# -------------------------------------------------------
# Vectorization + Model Selection
# -------------------------------------------------------
if model_choice == "NMF":

    vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=2000
    )

    X = vectorizer.fit_transform(processed_docs)

    model = NMF(
        n_components=num_topics,
        random_state=42
    )

else:

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=2000
    )

    X = vectorizer.fit_transform(processed_docs)

    model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42
    )

# -------------------------------------------------------
# Train Model
# -------------------------------------------------------
st.write("Training model...")
model.fit(X)

# -------------------------------------------------------
# Show Topics
# -------------------------------------------------------
feature_names = vectorizer.get_feature_names_out()

st.subheader("Top Words Per Topic")

for topic_idx, topic in enumerate(model.components_):
    words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    st.write(f"Topic {topic_idx+1}: " + " ".join(words))

# -------------------------------------------------------
# pyLDAvis Visualization (ONLY for LDA)
# -------------------------------------------------------
if model_choice == "LDA":

    st.subheader("📊 Interactive Topic Visualization")

    vis = pyLDAvis.lda_model.prepare(model, X, vectorizer)

    html_string = pyLDAvis.prepared_data_to_html(vis)

    components.html(
        html_string,
        height=800,
        scrolling=True
    )
