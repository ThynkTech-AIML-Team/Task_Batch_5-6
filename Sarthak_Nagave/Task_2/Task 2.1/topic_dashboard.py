import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess

import pyLDAvis
import pyLDAvis.lda_model

def display_topics(model, feature_names, model_name, n_words=10):
    print(f"\nTop words in {model_name}:")
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-n_words:]]
        print(f"Topic {topic_idx+1}: {', '.join(top_words)}")

def compute_coherence(model, feature_names, texts):
    topics = []
    for topic in model.components_:
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        topics.append(top_words)

    dictionary = Dictionary(texts)

    coherence_model = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v',
        processes=1 
    )

    return coherence_model.get_coherence()

def main():

    print("Loading 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    documents = dataset.data[:2000]   

    print("Vectorizing text...")

    count_vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    X_count = count_vectorizer.fit_transform(documents)
    X_tfidf = tfidf_vectorizer.fit_transform(documents)

    print("Training LDA model...")
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X_count)

    print("Training NMF model...")
    nmf = NMF(n_components=5, random_state=42)
    nmf.fit(X_tfidf)


    display_topics(lda, count_vectorizer.get_feature_names_out(), "LDA")
    display_topics(nmf, tfidf_vectorizer.get_feature_names_out(), "NMF")

    lda_perplexity = lda.perplexity(X_count)
    print("\nLDA Perplexity:", round(lda_perplexity, 2))

    print("Preparing coherence computation...")
    texts = [simple_preprocess(doc) for doc in documents]

    lda_coherence = compute_coherence(
        lda,
        count_vectorizer.get_feature_names_out(),
        texts
    )

    nmf_coherence = compute_coherence(
        nmf,
        tfidf_vectorizer.get_feature_names_out(),
        texts
    )

    print("LDA Coherence:", round(lda_coherence, 4))
    print("NMF Coherence:", round(nmf_coherence, 4))

    print("Generating pyLDAvis dashboard...")
    lda_vis = pyLDAvis.lda_model.prepare(
        lda,
        X_count,
        count_vectorizer
    )

    pyLDAvis.save_html(lda_vis, "lda_dashboard.html")

    print("\nDashboard saved as 'lda_dashboard.html'")
    print("Open this file in your browser to view interactive topics.")

if __name__ == "__main__":
    main()
