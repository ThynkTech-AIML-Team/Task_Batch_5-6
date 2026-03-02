import re
import numpy as np
import pandas as pd
import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
import nltk

# --- SETUP: FIX WINDOWS MULTIPROCESSING ERROR ---
# This protects the code from crashing on Windows when using multiple cores
if __name__ == '__main__':

    # 1. DOWNLOAD RESOURCES
    print("Downloading NLTK resources...")
    nltk.download('stopwords', quiet=True)
    stop_words = stopwords.words('english')
    # Add common useless words to stoplist
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    # [cite_start]2. LOAD DATA [cite: 3]
    print("Loading 20 Newsgroups dataset...")
    # Removing headers/footers ensures the model learns topics, not email metadata
    newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    data = newsgroups.data[:2000] # Use 2000 docs for faster processing
    print(f"Loaded {len(data)} documents.")

    # 3. PREPROCESSING
    def preprocess(text):
        # Remove emails
        text = re.sub(r'\S*@\S*\s?', '', text)
        # Remove newlines and extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove single quotes
        text = re.sub(r"\'", "", text)
        return text

    print("Cleaning and tokenizing data...")
    data_clean = [preprocess(doc) for doc in data]

    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    data_words = list(sent_to_words(data_clean))

    # Remove Stop Words
    data_words_nostops = [[word for word in doc if word not in stop_words] for doc in data_words]

    # 4. PREPARE DICTIONARY & CORPUS (For LDA)
    # Create Dictionary
    id2word = corpora.Dictionary(data_words_nostops)
    # Create Corpus
    corpus = [id2word.doc2bow(text) for text in data_words_nostops]

    # [cite_start]5. BUILD LDA MODEL (Latent Dirichlet Allocation) [cite: 4]
    print("Training LDA Model (this may take a minute)...")
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True)

    # Print the Keyword in the 10 topics
    print("\n--- LDA TOPICS ---")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic: {idx} \nWords: {topic}\n")

    # [cite_start]6. MODEL EVALUATION [cite: 6]
    # Compute Perplexity
    print(f'\nPerplexity: {lda_model.log_perplexity(corpus)}')  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=data_words_nostops, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}')

    # [cite_start]7. BUILD NMF MODEL (Non-Negative Matrix Factorization) [cite: 4]
    print("\nTraining NMF Model...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(data_clean)
    nmf = NMF(n_components=10, random_state=1, init='nndsvd').fit(tfidf)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    print("\n--- NMF TOPICS ---")
    for topic_idx, topic in enumerate(nmf.components_):
        print(f"Topic {topic_idx}: " + " ".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))

    # [cite_start]8. VISUALIZATION DASHBOARD [cite: 5]
    print("\nGenerating Dashboard...")
    # Enable notebook mode if using Jupyter, else save to HTML
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    output_filename = 'lda_visualization.html'
    pyLDAvis.save_html(vis, output_filename)
    print(f"Dashboard saved as '{output_filename}'. Open this file in your browser to see the bubbles!")