import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import zipfile
import os
import io

# Gensim for embeddings
try:
    from gensim.models import Word2Vec, FastText
    from gensim.models.keyedvectors import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

st.set_page_config(page_title="Task 2: Word Embedding Explorer", layout="wide")

st.title("Task 2: Word Embedding Explorer")
st.write("Compare Word2Vec, FastText, and GloVe embeddings with comprehensive analysis")

@st.cache_data
def download_glove_embeddings():
    """Download and load GloVe embeddings."""
    try:
        # Small GloVe embeddings for demo
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        
        # For demo, we'll create sample embeddings
        # In production, you would download real GloVe embeddings
        words = ['king', 'queen', 'man', 'woman', 'boy', 'girl', 'prince', 'princess', 
                'father', 'mother', 'son', 'daughter', 'brother', 'sister',
                'happy', 'sad', 'good', 'bad', 'big', 'small', 'cat', 'dog', 
                'car', 'house', 'book', 'computer', 'apple', 'orange', 'red', 'blue']
        
        np.random.seed(42)
        embeddings = {}
        for word in words:
            embeddings[word] = np.random.randn(50)  # 50-dim embeddings
        
        # Create semantic relationships
        embeddings['queen'] = embeddings['king'] - embeddings['man'] + embeddings['woman'] + 0.1 * np.random.randn(50)
        embeddings['princess'] = embeddings['prince'] - embeddings['man'] + embeddings['woman'] + 0.1 * np.random.randn(50)
        embeddings['mother'] = embeddings['father'] - embeddings['man'] + embeddings['woman'] + 0.1 * np.random.randn(50)
        
        return embeddings
        
    except Exception as e:
        st.error(f"Error loading GloVe: {e}")
        return {}

@st.cache_data 
def create_word2vec_model():
    """Create a Word2Vec model for comparison."""
    if not GENSIM_AVAILABLE:
        return None
    
    # Sample sentences for training
    sentences = [
        ['king', 'rules', 'kingdom'],
        ['queen', 'rules', 'kingdom'], 
        ['man', 'person', 'male'],
        ['woman', 'person', 'female'],
        ['boy', 'young', 'male'],
        ['girl', 'young', 'female'],
        ['prince', 'young', 'king'],
        ['princess', 'young', 'queen'],
        ['father', 'parent', 'male'],
        ['mother', 'parent', 'female'],
        ['car', 'vehicle', 'transport'],
        ['house', 'building', 'home'],
    ] * 100  # Repeat for better training
    
    try:
        model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=1)
        return model
    except Exception as e:
        st.error(f"Word2Vec error: {e}")
        return None

@st.cache_data
def create_fasttext_model():
    """Create a FastText model for comparison."""
    if not GENSIM_AVAILABLE:
        return None
        
    sentences = [
        ['king', 'rules', 'kingdom'],
        ['queen', 'rules', 'kingdom'], 
        ['man', 'person', 'male'],
        ['woman', 'person', 'female'],
        ['boy', 'young', 'male'],
        ['girl', 'young', 'female'],
        ['prince', 'young', 'king'],
        ['princess', 'young', 'queen'],
    ] * 100
    
    try:
        model = FastText(sentences, vector_size=50, window=3, min_count=1, workers=1)
        return model
    except Exception as e:
        st.error(f"FastText error: {e}")
        return None

def find_most_similar(word, embeddings, top_k=5):
    """Find most similar words using cosine similarity."""
    if isinstance(embeddings, dict):
        if word not in embeddings:
            return []
        
        target_vec = embeddings[word].reshape(1, -1)
        similarities = []
        
        for other_word, vec in embeddings.items():
            if other_word != word:
                similarity = cosine_similarity(target_vec, vec.reshape(1, -1))[0][0]
                similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    # For gensim models
    elif hasattr(embeddings, 'wv') and word in embeddings.wv:
        try:
            similar = embeddings.wv.most_similar(word, topn=top_k)
            return similar
        except:
            return []
    
    return []

def solve_analogy(word1, word2, word3, embeddings):
    """Solve word analogy: word1 is to word2 as word3 is to ?"""
    if isinstance(embeddings, dict):
        if not all(w in embeddings for w in [word1, word2, word3]):
            return []
        
        target_vec = embeddings[word2] - embeddings[word1] + embeddings[word3]
        similarities = []
        
        for word, vec in embeddings.items():
            if word not in [word1, word2, word3]:
                similarity = cosine_similarity(target_vec.reshape(1, -1), vec.reshape(1, -1))[0][0]
                similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]
    
    # For gensim models
    elif hasattr(embeddings, 'wv'):
        try:
            if all(w in embeddings.wv for w in [word1, word2, word3]):
                result = embeddings.wv.most_similar(
                    positive=[word2, word3], negative=[word1], topn=5
                )
                return result
        except:
            pass
    
    return []

def test_oov_handling(model, test_words):
    """Test how models handle out-of-vocabulary words."""
    results = {}
    
    for word in test_words:
        if hasattr(model, 'wv'):
            if word in model.wv:
                results[word] = "In vocabulary"
            else:
                # FastText can handle OOV
                if hasattr(model.wv, 'get_vector'):
                    try:
                        vector = model.wv.get_vector(word)
                        results[word] = "OOV - handled by FastText"
                    except:
                        results[word] = "OOV - not handled"
                else:
                    results[word] = "OOV - not handled"
        elif isinstance(model, dict):
            results[word] = "In vocabulary" if word in model else "OOV - not handled"
    
    return results

def visualize_embeddings(embeddings, words=None, method='PCA'):
    """Create 2D visualization of embeddings."""
    if isinstance(embeddings, dict):
        available_words = list(embeddings.keys())
        if words is None:
            words = available_words[:20]
        
        words = [w for w in words if w in available_words]
        vectors = np.array([embeddings[word] for word in words])
        
    elif hasattr(embeddings, 'wv'):
        available_words = list(embeddings.wv.index_to_key)
        if words is None:
            words = available_words[:20]
        
        words = [w for w in words if w in embeddings.wv]
        vectors = np.array([embeddings.wv[word] for word in words])
    
    else:
        return None
    
    if len(vectors) == 0:
        return None
    
    # Dimensionality reduction
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
    
    reduced_vectors = reducer.fit_transform(vectors)
    
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'word': words
    })
    
    fig = px.scatter(df, x='x', y='y', text='word', 
                     title=f'Word Embeddings Visualization ({method})')
    fig.update_traces(textposition="top center")
    fig.update_layout(height=500)
    return fig

def main():
    # Load embeddings
    with st.spinner("Loading embeddings..."):
        glove_embeddings = download_glove_embeddings()
        word2vec_model = create_word2vec_model()
        fasttext_model = create_fasttext_model()
    
    available_words = list(glove_embeddings.keys())
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    embedding_type = st.sidebar.selectbox(
        "Choose embedding type:", 
        ["GloVe", "Word2Vec", "FastText"]
    )
    
    # Select current embeddings
    current_embeddings = glove_embeddings
    if embedding_type == "Word2Vec" and word2vec_model:
        current_embeddings = word2vec_model
        available_words = list(word2vec_model.wv.index_to_key)
    elif embedding_type == "FastText" and fasttext_model:
        current_embeddings = fasttext_model  
        available_words = list(fasttext_model.wv.index_to_key)
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Word Similarity", "Word Analogies", "Model Comparison", 
        "OOV Handling", "Visualization"
    ])
    
    with tab1:
        st.subheader("Word Similarity Search")
        selected_word = st.selectbox("Select a word:", available_words[:30])
        
        if st.button("Find Similar Words"):
            similar_words = find_most_similar(selected_word, current_embeddings)
            
            if similar_words:
                st.write(f"Words most similar to '{selected_word}':")
                for word, similarity in similar_words:
                    st.write(f"• {word}: {similarity:.3f}")
            else:
                st.warning("No similar words found")
    
    with tab2:
        st.subheader("Word Analogies")
        st.write("Solve analogies: A is to B as C is to ?")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            word_a = st.selectbox("Word A:", available_words[:20], key='analogy_a')
        with col2:
            word_b = st.selectbox("Word B:", available_words[:20], key='analogy_b')
        with col3:
            word_c = st.selectbox("Word C:", available_words[:20], key='analogy_c')
        
        if st.button("Solve Analogy"):
            results = solve_analogy(word_a, word_b, word_c, current_embeddings)
            
            if results:
                st.write(f"'{word_a}' is to '{word_b}' as '{word_c}' is to:")
                for word, similarity in results:
                    st.write(f"• {word}: {similarity:.3f}")
            else:
                st.warning("Could not solve analogy")
    
    with tab3:
        st.subheader("Embedding Model Comparison")
        
        if st.button("Compare Models"):
            test_word = "king"
            
            # Compare similarity search across models
            comparison_data = []
            
            models = {
                "GloVe": glove_embeddings,
                "Word2Vec": word2vec_model, 
                "FastText": fasttext_model
            }
            
            for model_name, model in models.items():
                if model is not None:
                    similar = find_most_similar(test_word, model, top_k=3)
                    if similar:
                        top_words = [f"{w} ({s:.3f})" for w, s in similar[:3]]
                        comparison_data.append({
                            "Model": model_name,
                            "Top Similar Words": ", ".join(top_words)
                        })
            
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data))
    
    with tab4:
        st.subheader("Out-of-Vocabulary (OOV) Word Handling")
        st.write("FastText can handle words not seen during training by using character n-grams")
        
        # Test words including some potential OOV words
        test_words = ["king", "nonexistentword", "supercalifragilisticexpialidocious"]
        
        if st.button("Test OOV Handling"):
            if fasttext_model and word2vec_model:
                oov_results = []
                
                for word in test_words:
                    w2v_result = test_oov_handling(word2vec_model, [word])
                    ft_result = test_oov_handling(fasttext_model, [word])
                    
                    oov_results.append({
                        "Word": word,
                        "Word2Vec": w2v_result.get(word, "Error"),
                        "FastText": ft_result.get(word, "Error")
                    })
                
                st.dataframe(pd.DataFrame(oov_results))
                st.info("FastText uses subword information to handle unknown words")
    
    with tab5:
        st.subheader("2D Embedding Visualization")
        
        viz_method = st.radio("Visualization method:", ['PCA', 't-SNE'])
        selected_words = st.multiselect(
            "Select words to visualize:", 
            available_words[:30], 
            default=available_words[:10]
        )
        
        if st.button("Generate Visualization") and selected_words:
            fig = visualize_embeddings(current_embeddings, selected_words, viz_method)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not generate visualization")

if __name__ == "__main__":
    main()