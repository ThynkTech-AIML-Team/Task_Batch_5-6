import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re

# NLP libraries with Python 3.14 compatibility
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

st.set_page_config(page_title="End-to-End NLP Applications", layout="wide")

st.title("End-to-End NLP Applications")
st.write("Production-ready NLP tools for real-world applications")

@st.cache_resource
def load_sentiment_models():
    """Load multilingual sentiment analysis models."""
    models = {}
    try:
        # English sentiment analysis
        models['english'] = pipeline("sentiment-analysis", 
                                   model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Multilingual sentiment analysis
        models['multilingual'] = pipeline("sentiment-analysis", 
                                        model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        return models
    except Exception as e:
        st.error(f"Error loading sentiment models: {e}")
        return {}

@st.cache_resource 
def load_spacy_model():
    """Load spaCy model for NER."""
    if not SPACY_AVAILABLE:
        return None
    try:
        # Try to load English model
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except Exception as e:
        st.warning(f"spaCy model not available: {e}")
        return None

@st.cache_data
def load_sample_news():
    """Load sample news articles for similarity search."""
    return [
        "Scientists discover breakthrough in quantum computing technology that could revolutionize data processing",
        "Stock market reaches record highs as technology companies show strong quarterly earnings",
        "New artificial intelligence system demonstrates human-level performance in medical diagnosis",
        "Climate change impacts becoming more severe with rising global temperatures and extreme weather",
        "Major technology companies announce massive investments in renewable energy infrastructure",
        "Space exploration mission successfully lands on Mars and begins collecting geological samples",
        "Medical researchers develop promising new treatment approach for cancer using immunotherapy",
        "Economic indicators suggest steady recovery following recent market volatility and uncertainty",
        "Machine learning algorithms improve educational outcomes through personalized learning systems",
        "Environmental protection efforts show positive results in biodiversity conservation programs"
    ]

def news_similarity_search():
    """News similarity search application."""
    st.subheader("News Similarity Search")
    st.write("Find articles similar to your input query")
    
    news_articles = load_sample_news()
    
    # Display available articles
    with st.expander("Available News Articles", expanded=False):
        for i, article in enumerate(news_articles, 1):
            st.write(f"{i}. {article}")
    
    # User input
    user_query = st.text_area("Enter your search query or paste an article:", 
                             placeholder="Type your query here...")
    
    if st.button("Find Similar Articles", type="primary") and user_query.strip():
        # Vectorize texts
        all_texts = news_articles + [user_query]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarities
        query_vector = tfidf_matrix[-1]
        article_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vector, article_vectors)[0]
        
        # Get top 3 similar articles
        similar_indices = np.argsort(similarities)[::-1][:3]
        
        st.subheader("Top 3 Most Similar Articles:")
        for i, idx in enumerate(similar_indices, 1):
            similarity_score = similarities[idx]
            if similarity_score > 0:
                st.write(f"**{i}. Similarity: {similarity_score:.3f}**")
                st.write(news_articles[idx])
                st.write("---")
            else:
                st.write(f"**{i}. Similarity: {similarity_score:.3f}**")
                st.write("No significant similarity found")

def resume_parser_with_ner():
    """Resume parser using Named Entity Recognition."""
    st.subheader("Resume Parser with NER")
    st.write("Extract skills, education, organizations, and names from resumes")
    
    nlp = load_spacy_model()
    
    # Sample resume text
    sample_resume = """
    John Smith
    Software Engineer
    Email: john.smith@email.com
    Phone: (123) 456-7890
    
    EXPERIENCE
    Software Developer at Google Inc. (2020-2023)
    - Developed machine learning algorithms using Python and TensorFlow
    - Worked with cloud computing platforms like AWS and Azure
    - Led a team of 5 engineers on microservices architecture
    
    Data Analyst at Microsoft Corporation (2018-2020)
    - Analyzed large datasets using SQL and pandas
    - Created data visualizations with matplotlib and plotly
    - Collaborated with product managers and designers
    
    EDUCATION
    Master of Science in Computer Science, Stanford University (2016-2018)
    Bachelor of Science in Mathematics, MIT (2012-2016)
    
    SKILLS
    Programming: Python, Java, JavaScript, C++
    Frameworks: React, Django, Flask, Spring Boot
    Databases: MySQL, PostgreSQL, MongoDB
    Tools: Git, Docker, Kubernetes, Jenkins
    """
    
    resume_text = st.text_area("Paste resume text here:", value=sample_resume, height=300)
    
    if st.button("Parse Resume", type="primary") and resume_text.strip():
        
        if nlp is not None:
            # Process with spaCy
            doc = nlp(resume_text)
            
            # Extract entities
            entities = {
                "PERSON": [],
                "ORG": [],
                "GPE": [],  # Geographic entities
                "DATE": [],
                "MONEY": []
            }
            
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
            
            # Extract skills using pattern matching
            skills_patterns = [
                r'\b(Python|Java|JavaScript|C\+\+|SQL|HTML|CSS|React|Django|Flask)\b',
                r'\b(AWS|Azure|GCP|Docker|Kubernetes|Git|Jenkins)\b',
                r'\b(MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch)\b',
                r'\b(TensorFlow|PyTorch|scikit-learn|pandas|numpy)\b'
            ]
            
            extracted_skills = set()
            for pattern in skills_patterns:
                matches = re.findall(pattern, resume_text, re.IGNORECASE)
                extracted_skills.update(matches)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Named Entities")
                for entity_type, values in entities.items():
                    if values:
                        st.write(f"**{entity_type}:**")
                        for value in set(values):
                            st.write(f"• {value}")
                        st.write("")
            
            with col2:
                st.subheader("Extracted Skills")
                if extracted_skills:
                    for skill in sorted(extracted_skills):
                        st.write(f"• {skill}")
                else:
                    st.write("No technical skills detected")
                    
                st.subheader("Education & Organizations")
                orgs = set(entities["ORG"])
                if orgs:
                    st.write("**Organizations:**")
                    for org in orgs:
                        st.write(f"• {org}")
                        
        else:
            # Fallback without spaCy
            st.warning("spaCy not available. Using pattern-based extraction.")
            
            # Simple pattern-based extraction
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            
            emails = re.findall(email_pattern, resume_text)
            phones = re.findall(phone_pattern, resume_text)
            
            st.write("**Contact Information:**")
            if emails:
                st.write(f"Email: {emails[0]}")
            if phones:
                st.write(f"Phone: {phones[0]}")

def multilingual_sentiment_analysis():
    """Multilingual sentiment analysis application."""
    st.subheader("Multilingual Sentiment Analysis")
    st.write("Compare sentiment analysis across English and other languages")
    
    sentiment_models = load_sentiment_models()
    
    # Sample texts in different languages
    sample_texts = {
        "English": "I love this product! It works perfectly and exceeded my expectations.",
        "Spanish": "Me encanta este producto! Funciona perfectamente y superó mis expectativas.",
        "French": "J'adore ce produit! Il fonctionne parfaitement et a dépassé mes attentes.",
        "German": "Ich liebe dieses Produkt! Es funktioniert perfekt und hat meine Erwartungen übertroffen."
    }
    
    # Language selection
    language = st.selectbox("Select language:", list(sample_texts.keys()))
    
    # Text input
    text_input = st.text_area("Enter text for sentiment analysis:", 
                             value=sample_texts[language], 
                             height=100)
    
    if st.button("Analyze Sentiment", type="primary") and text_input.strip():
        results = []
        
        # Analyze with both models if available
        for model_name, model in sentiment_models.items():
            if model:
                try:
                    result = model(text_input)[0]
                    results.append({
                        "Model": model_name,
                        "Sentiment": result['label'],
                        "Confidence": f"{result['score']:.3f}"
                    })
                except Exception as e:
                    results.append({
                        "Model": model_name,
                        "Sentiment": f"Error: {e}",
                        "Confidence": "N/A"
                    })
        
        # Display results
        if results:
            st.subheader("Sentiment Analysis Results")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results)
            
            # Visual representation
            for result in results:
                if "Error" not in result["Sentiment"]:
                    confidence = float(result["Confidence"])
                    sentiment = result["Sentiment"]
                    
                    if sentiment in ["POSITIVE", "LABEL_2"]:
                        st.success(f"{result['Model']}: Positive sentiment (confidence: {confidence:.3f})")
                    elif sentiment in ["NEGATIVE", "LABEL_0"]:
                        st.error(f"{result['Model']}: Negative sentiment (confidence: {confidence:.3f})")
                    else:
                        st.info(f"{result['Model']}: Neutral sentiment (confidence: {confidence:.3f})")

def main():
    # Sidebar navigation
    st.sidebar.header("Choose Application")
    app_choice = st.sidebar.selectbox(
        "Select NLP Application:",
        [
            "News Similarity Search",
            "Resume Parser with NER", 
            "Multilingual Sentiment Analysis"
        ]
    )
    
    # Run selected application
    if app_choice == "News Similarity Search":
        news_similarity_search()
    elif app_choice == "Resume Parser with NER":
        resume_parser_with_ner()
    elif app_choice == "Multilingual Sentiment Analysis":
        multilingual_sentiment_analysis()

if __name__ == "__main__":
    main()