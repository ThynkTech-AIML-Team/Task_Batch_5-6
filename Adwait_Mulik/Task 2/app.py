import streamlit as st
import pyLDAvis
import pyLDAvis.lda_model
import streamlit.components.v1 as components
from src.topic_modeling import train_models

st.set_page_config(page_title="Topic Modeling Dashboard", layout="wide")
st.title("🔬 Topic Modeling Research Dashboard")

if st.sidebar.button("Train and Visualize Models"):
    with st.spinner("Training models... Please wait."):
        lda_model, nmf_model, tf_vectorizer, tf = train_models()
        
        # Prepare pyLDAvis visualization
        vis_data = pyLDAvis.lda_model.prepare(lda_model, tf, tf_vectorizer)
        html_obj = pyLDAvis.prepared_data_to_html(vis_data)
        
        st.success("Models Trained Successfully!")
        
        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("LDA Perplexity", f"{lda_model.perplexity(tf):.2f}")
        
        st.subheader("Interactive Topic Visualization (pyLDAvis)")
        components.html(html_obj, height=800, scrolling=True)

else:
    st.info("Click the button in the sidebar to start the Topic Modeling process.")