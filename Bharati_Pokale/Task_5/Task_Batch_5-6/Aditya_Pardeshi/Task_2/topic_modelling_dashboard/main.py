from sklearn.datasets import fetch_20newsgroups
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pyLDAvis
from pyLDAvis import prepare
import numpy as np


def preprocess_documents(documents):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        tokens = text.split()
        tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stop_words and len(word) > 3
        ]
        return " ".join(tokens)

    print("Preprocessing documents...")
    processed_docs = [preprocess(doc) for doc in documents]
    return processed_docs


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx}:")
        print([
            feature_names[i]
            for i in topic.argsort()[:-no_top_words - 1:-1]
        ])

def compute_coherence_values(processed_docs, start=5, limit=20, step=5):

        texts = [doc.split() for doc in processed_docs]
        dictionary = Dictionary(texts)
        dictionary.filter_extremes(no_below=5, no_above=0.9)

        vectorizer = CountVectorizer(max_df=0.9, min_df=5)
        doc_term_matrix = vectorizer.fit_transform(processed_docs)
        feature_names = vectorizer.get_feature_names_out()

        coherence_values = []
        topic_numbers = []

        for num_topics in range(start, limit, step):
            print(f"\nTraining LDA with {num_topics} topics...")

            lda_model = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42
            )

            lda_model.fit(doc_term_matrix)

            lda_topics = [
                [feature_names[i] for i in topic.argsort()[:-11:-1]]
                for topic in lda_model.components_
            ]

            coherence_model = CoherenceModel(
                topics=lda_topics,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )

            coherence = coherence_model.get_coherence()
            print(f"Coherence Score for {num_topics}: {coherence}")

            coherence_values.append(coherence)
            topic_numbers.append(num_topics)

        return topic_numbers, coherence_values



def main():

    print("Loading dataset...")
    dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    documents = dataset.data
    print("Total documents:", len(documents))

    # Preprocess
    processed_docs = preprocess_documents(documents)

    # Create Document-Term Matrix
    print("Creating document-term matrix...")
    vectorizer = CountVectorizer(max_df=0.9, min_df=5)
    doc_term_matrix = vectorizer.fit_transform(processed_docs)
    print("Matrix shape:", doc_term_matrix.shape)

    # Train LDA
    print("Training LDA model...")
    num_topics = 10
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42
    )
    lda_model.fit(doc_term_matrix)
    print("LDA training complete.")

    # Display Topics
    feature_names = vectorizer.get_feature_names_out()
    display_topics(lda_model, feature_names, 10)

    # Perplexity
    perplexity = lda_model.perplexity(doc_term_matrix)
    print("\nPerplexity:", perplexity)

    # Coherence Calculation
    print("\nPreparing data for coherence score...")

    texts = [doc.split() for doc in processed_docs]
    dictionary = Dictionary(texts)

    # filter dictionary similar to vectorizer
    dictionary.filter_extremes(no_below=5, no_above=0.9)

    lda_topics = [
        [feature_names[i] for i in topic.argsort()[:-11:-1]]
        for topic in lda_model.components_
    ]

    coherence_model = CoherenceModel(
        topics=lda_topics,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()
    print("Coherence Score:", coherence_score)
    
    print("\nPreparing LDA visualization...")

    # Topic-word distribution
    topic_term_dists = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]

    # Document-topic distribution
    doc_topic_dists = lda_model.transform(doc_term_matrix)

    # Document lengths
    doc_lengths = [len(doc.split()) for doc in processed_docs]

    # Vocabulary
    vocab = vectorizer.get_feature_names_out()

    # Term frequencies
    term_frequency = np.array(doc_term_matrix.sum(axis=0)).flatten()

    lda_vis = pyLDAvis.prepare(
        topic_term_dists,
        doc_topic_dists,
        doc_lengths,
        vocab,
        term_frequency
    )

    pyLDAvis.save_html(lda_vis, "lda_visualization.html")

    print("LDA visualization saved as lda_visualization.html")
    
    print("\nFinding optimal number of topics...")

    topic_numbers, coherence_values = compute_coherence_values(processed_docs)

    print("\nTopic Numbers:", topic_numbers)
    print("Coherence Values:", coherence_values)

    plt.plot(topic_numbers, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence vs Number of Topics")
    plt.show()

    #NMF Section
    print("\nCreating TF-IDF matrix for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=5)
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

    print("\nTraining NMF model...")
    nmf_model = NMF(
        n_components=10,
        random_state=42
    )
    nmf_model.fit(tfidf_matrix)
    print("NMF training complete.")

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    print("\nNMF Topics:")
    display_topics(nmf_model, tfidf_feature_names, 10)

    #NMF Cohenrance
    nmf_topics = [
        [tfidf_feature_names[i] for i in topic.argsort()[:-11:-1]]
        for topic in nmf_model.components_
    ]

    nmf_coherence_model = CoherenceModel(
        topics=nmf_topics,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )

    nmf_coherence = nmf_coherence_model.get_coherence()
    print("\nNMF Coherence Score:", nmf_coherence)

    print("\n===== MODEL COMPARISON =====")
    print(f"LDA Coherence: {coherence_score}")
    print(f"NMF Coherence: {nmf_coherence}")
    print(f"LDA Perplexity: {perplexity}")




if __name__ == "__main__":
    main()
