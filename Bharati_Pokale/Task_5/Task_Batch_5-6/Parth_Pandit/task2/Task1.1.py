

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

import pyLDAvis
import pyLDAvis.lda_model

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

try:
    from bertopic import BERTopic
    bertopic_available = True
except ImportError:
    bertopic_available = False


def main():


    data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    documents = data.data
    print("Total documents:", len(documents))


    count_vectorizer = CountVectorizer(
        stop_words='english',
        max_df=0.95,
        min_df=2
    )
    count_data = count_vectorizer.fit_transform(documents)

    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.95,
        min_df=2
    )
    tfidf_data = tfidf_vectorizer.fit_transform(documents)


    lda_model = LatentDirichletAllocation(
        n_components=10,
        random_state=42
    )
    lda_model.fit(count_data)

    nmf_model = NMF(
        n_components=10,
        random_state=42
    )
    nmf_model.fit(tfidf_data)


    def extract_topics(model, feature_names, num_words=10):
        topics = []
        for topic in model.components_:
            words = [feature_names[i] for i in topic.argsort()[:-num_words-1:-1]]
            topics.append(words)
        return topics

    lda_topics = extract_topics(
        lda_model,
        count_vectorizer.get_feature_names_out()
    )

    nmf_topics = extract_topics(
        nmf_model,
        tfidf_vectorizer.get_feature_names_out()
    )

    print("\nLDA Topics:")
    for i, t in enumerate(lda_topics):
        print(f"Topic {i+1}: {t}")

    print("\nNMF Topics:")
    for i, t in enumerate(nmf_topics):
        print(f"Topic {i+1}: {t}")


    lda_perplexity = lda_model.perplexity(count_data)
    print("\nLDA Perplexity:", lda_perplexity)


    texts = [doc.split() for doc in documents]
    dictionary = Dictionary(texts)

    lda_coherence = CoherenceModel(
        topics=lda_topics,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v',
        processes=1      # 🔥 KEY FIX
    ).get_coherence()

    nmf_coherence = CoherenceModel(
        topics=nmf_topics,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v',
        processes=1      # 🔥 KEY FIX
    ).get_coherence()

    print("LDA Coherence Score:", lda_coherence)
    print("NMF Coherence Score:", nmf_coherence)

    lda_vis = pyLDAvis.lda_model.prepare(
        lda_model,
        count_data,
        count_vectorizer
    )

    pyLDAvis.save_html(lda_vis, "lda_dashboard.html")
    print("pyLDAvis dashboard saved as lda_dashboard.html")

    comparison = pd.DataFrame({
        "Model": ["LDA", "NMF"],
        "Coherence Score": [lda_coherence, nmf_coherence],
        "Perplexity": [lda_perplexity, "Not Applicable"]
    })

    print("\nModel Comparison:")
    print(comparison)


    if bertopic_available:
        bertopic_model = BERTopic()
        topics, probs = bertopic_model.fit_transform(documents)
        print("\nBERTopic Top Topics:")
        print(bertopic_model.get_topic_info().head())
    else:
        print("\nBERTopic not installed. Bonus skipped.")



if __name__ == "__main__":
    main()
