import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Advanced topic modeling imports
try:
    import pyLDAvis
    import pyLDAvis.sklearn
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

try:
    import gensim
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

st.set_page_config(page_title="Task 1: Topic Modeling Research Dashboard", layout="wide")

st.title("Task 1: Topic Modeling Research Dashboard")
st.write("Compare LDA, NMF, and BERTopic models with comprehensive evaluation")

@st.cache_data
def load_data():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    return newsgroups.data[:1000], newsgroups.target_names

@st.cache_data
def preprocess_data(texts):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', min_df=2, max_df=0.8)
    doc_term_matrix = vectorizer.fit_transform(texts)
    return doc_term_matrix, vectorizer

def train_lda_model(doc_term_matrix, n_topics):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda.fit(doc_term_matrix)
    return lda

def train_nmf_model(doc_term_matrix, n_topics):
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=100)
    nmf.fit(doc_term_matrix)
    return nmf

def train_bertopic_model(texts, n_topics):
    if not BERTOPIC_AVAILABLE:
        return None
    try:
        topic_model = BERTopic(nr_topics=n_topics, verbose=False)
        topics, probs = topic_model.fit_transform(texts)
        return topic_model, topics, probs
    except Exception as e:
        st.error(f"BERTopic error: {e}")
        return None

def calculate_coherence(model, doc_term_matrix, texts, vectorizer):
    if not GENSIM_AVAILABLE:
        return None
    try:
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic in model.components_:
            topic_words = [feature_names[i] for i in topic.argsort()[-10:]]
            topics.append(topic_words)
        
        # Prepare texts for gensim
        processed_texts = [text.split() for text in texts[:100]]  # Sample for speed
        dictionary = Dictionary(processed_texts)
        
        cm = CoherenceModel(topics=topics, texts=processed_texts, 
                           dictionary=dictionary, coherence='c_v')
        return cm.get_coherence()
    except Exception as e:
        return None

def display_topics(model, feature_names, no_top_words=8):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return topics

def create_topic_visualization(model, doc_term_matrix, vectorizer):
    if not PYLDAVIS_AVAILABLE:
        return None
    try:
        vis = pyLDAvis.sklearn.prepare(model, doc_term_matrix, vectorizer, mds='tsne')
        return vis
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return None

def main():
    texts, categories = load_data()
    st.write(f"Dataset: {len(texts)} documents from 20 newsgroups")
    
    # Sidebar controls
    st.sidebar.header("Model Configuration")
    n_topics = st.sidebar.slider("Number of topics", 2, 15, 8)
    model_types = st.sidebar.multiselect("Select models to train", 
                                       ["LDA", "NMF", "BERTopic"], 
                                       default=["LDA", "NMF"])
    
    # Preprocess data
    doc_term_matrix, vectorizer = preprocess_data(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    if st.sidebar.button("Train Selected Models", type="primary"):
        results = {}
        
        with st.spinner("Training models..."):
            
            # Train LDA
            if "LDA" in model_types:
                lda_model = train_lda_model(doc_term_matrix, n_topics)
                lda_perplexity = lda_model.perplexity(doc_term_matrix)
                lda_coherence = calculate_coherence(lda_model, doc_term_matrix, texts[:100], vectorizer)
                results["LDA"] = {
                    "model": lda_model, 
                    "perplexity": lda_perplexity,
                    "coherence": lda_coherence
                }
            
            # Train NMF  
            if "NMF" in model_types:
                nmf_model = train_nmf_model(doc_term_matrix, n_topics)
                nmf_coherence = calculate_coherence(nmf_model, doc_term_matrix, texts[:100], vectorizer)
                results["NMF"] = {
                    "model": nmf_model,
                    "perplexity": None,
                    "coherence": nmf_coherence
                }
            
            # Train BERTopic
            if "BERTopic" in model_types and BERTOPIC_AVAILABLE:
                bertopic_result = train_bertopic_model(texts[:200], n_topics)
                if bertopic_result:
                    topic_model, topics, probs = bertopic_result
                    results["BERTopic"] = {
                        "model": topic_model,
                        "topics": topics,
                        "probs": probs
                    }
        
        st.session_state.results = results
        st.session_state.feature_names = feature_names
        st.session_state.doc_term_matrix = doc_term_matrix
        st.session_state.vectorizer = vectorizer
        st.success("Models trained successfully!")
    
    # Display results
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Model comparison metrics
        st.subheader("Model Comparison")
        
        metrics_data = []
        for model_name, result in results.items():
            if model_name != "BERTopic":
                metrics_data.append({
                    "Model": model_name,
                    "Perplexity": result.get("perplexity", "N/A"),
                    "Coherence": result.get("coherence", "N/A")
                })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics)
        
        # Topic displays
        st.subheader("Discovered Topics")
        
        tab_names = list(results.keys())
        if tab_names:
            tabs = st.tabs(tab_names)
            
            for i, (model_name, result) in enumerate(results.items()):
                with tabs[i]:
                    if model_name == "BERTopic" and "model" in result:
                        try:
                            topic_info = result["model"].get_topic_info()
                            st.write("BERTopic Results:")
                            st.dataframe(topic_info.head(10))
                        except Exception as e:
                            st.error(f"BERTopic display error: {e}")
                    else:
                        topics = display_topics(result["model"], st.session_state.feature_names)
                        for topic in topics:
                            st.write(f"• {topic}")
        
        # Visualization section
        if PYLDAVIS_AVAILABLE and "LDA" in results:
            st.subheader("Interactive Topic Visualization (pyLDAvis)")
            if st.button("Generate LDA Visualization"):
                with st.spinner("Creating visualization..."):
                    vis = create_topic_visualization(
                        results["LDA"]["model"], 
                        st.session_state.doc_term_matrix, 
                        st.session_state.vectorizer
                    )
                    if vis:
                        html_string = pyLDAvis.prepared_data_to_html(vis)
                        st.components.v1.html(html_string, height=800)
        elif not PYLDAVIS_AVAILABLE:
            st.info("Install pyLDAvis for interactive topic visualization")

if __name__ == "__main__":
    main()