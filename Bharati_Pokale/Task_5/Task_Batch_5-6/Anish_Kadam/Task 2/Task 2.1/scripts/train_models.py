from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

def vectorize_docs(documents):
    # CountVectorizer for LDA
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = count_vectorizer.fit_transform(documents)

    # TF-IDF for NMF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    return doc_term_matrix, tfidf_matrix, count_vectorizer, tfidf_vectorizer

def train_lda(doc_term_matrix, num_topics=10):
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(doc_term_matrix)
    return lda_model

def train_nmf(tfidf_matrix, num_topics=10):
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(tfidf_matrix)
    return nmf_model
