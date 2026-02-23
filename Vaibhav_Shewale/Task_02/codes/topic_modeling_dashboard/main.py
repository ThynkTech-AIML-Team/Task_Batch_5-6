import re
import numpy as np
import matplotlib.pyplot as plt
import pyLDAvis
from pyLDAvis import prepare
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# --- PREPROCESSING ---
def preprocess_documents(documents):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        tokens = text.split()
        return " ".join([
            lemmatizer.lemmatize(w) for w in tokens 
            if w not in stop_words and len(w) > 3
        ])

    print("Preprocessing documents...")
    return [preprocess(doc) for doc in documents]

# --- UTILITIES ---
def get_lda_topics(model, feature_names, n_top_words):
    return [[feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] 
            for topic in model.components_]

def display_topics(topics, model_name):
    print(f"\n--- {model_name} Topics ---")
    for idx, topic in enumerate(topics):
        print(f"Topic {idx}: {', '.join(topic)}")

# --- VISUALIZATION HELPER ---
def save_lda_vis(model, dtm, vectorizer, processed_docs, filename):
    # Normalize with epsilon to avoid div by zero
    topic_term_dists = model.components_ / (model.components_.sum(axis=1)[:, np.newaxis] + 1e-10)
    doc_topic_dists = model.transform(dtm)
    doc_lengths = [len(doc.split()) for doc in processed_docs]
    vocab = vectorizer.get_feature_names_out()
    term_freq = np.array(dtm.sum(axis=0)).flatten()

    vis_data = pyLDAvis.prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_freq)
    pyLDAvis.save_html(vis_data, filename)
    print(f"Visualization saved to {filename}")

# --- MAIN EXECUTION ---
def main():
    # 1. Load & Preprocess
    dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    processed_docs = preprocess_documents(dataset.data)
    texts = [doc.split() for doc in processed_docs]
    gensim_dict = Dictionary(texts)
    gensim_dict.filter_extremes(no_below=5, no_above=0.9)

    # 2. LDA Pipeline
    print("\n[LDA Phase]")
    count_vec = CountVectorizer(max_df=0.9, min_df=5)
    dtm_count = count_vec.fit_transform(processed_docs)
    
    lda = LatentDirichletAllocation(n_components=10, random_state=42, n_jobs=-1)
    lda.fit(dtm_count)
    
    lda_topic_words = get_lda_topics(lda, count_vec.get_feature_names_out(), 10)
    display_topics(lda_topic_words, "LDA")
    
    lda_coherence = CoherenceModel(topics=lda_topic_words, texts=texts, dictionary=gensim_dict, coherence='c_v').get_coherence()

    # 3. NMF Pipeline
    print("\n[NMF Phase]")
    tfidf_vec = TfidfVectorizer(max_df=0.9, min_df=5)
    dtm_tfidf = tfidf_vec.fit_transform(processed_docs)
    
    nmf = NMF(n_components=10, random_state=42)
    nmf.fit(dtm_tfidf)
    
    nmf_topic_words = get_lda_topics(nmf, tfidf_vec.get_feature_names_out(), 10)
    display_topics(nmf_topic_words, "NMF")
    
    nmf_coherence = CoherenceModel(topics=nmf_topic_words, texts=texts, dictionary=gensim_dict, coherence='c_v').get_coherence()

    # 4. Results & Visualization
    print("\n" + "="*30)
    print(f"LDA Coherence: {lda_coherence:.4f}")
    print(f"NMF Coherence: {nmf_coherence:.4f}")
    print(f"LDA Perplexity: {lda.perplexity(dtm_count):.2f}")
    print("="*30)

    save_lda_vis(lda, dtm_count, count_vec, processed_docs, "lda_visualization.html")

if __name__ == "__main__":
    main()