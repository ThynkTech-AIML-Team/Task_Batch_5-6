# ============================================================
# TASK 1: Topic Modeling Research Dashboard
# Using: pandas, matplotlib, pyLDAvis, sklearn
# Dataset: 20 Newsgroups
# Models: LDA and NMF
# Output: Topic tables, plots, and pyLDAvis dashboard
# ============================================================

# -----------------------------
# Import Libraries
# -----------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.sklearn

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

import nltk
nltk.download('stopwords')


# ============================================================
# STEP 1: Load Dataset
# ============================================================

print("\nLoading 20 Newsgroups dataset...")

dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

documents = dataset.data[:2000]

df = pd.DataFrame({
    "Document": documents
})

print("Dataset loaded successfully.")
print("Total documents:", len(df))


# ============================================================
# STEP 2: Text Vectorization
# ============================================================

print("\nVectorizing text...")

# Count Vectorizer for LDA
count_vectorizer = CountVectorizer(
    stop_words='english',
    max_df=0.95,
    min_df=2
)

# FIX for pyLDAvis compatibility
count_vectorizer.get_feature_names = count_vectorizer.get_feature_names_out

count_matrix = count_vectorizer.fit_transform(df['Document'])


# TF-IDF Vectorizer for NMF
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.95,
    min_df=2
)

tfidf_matrix = tfidf_vectorizer.fit_transform(df['Document'])

print("Vectorization complete.")


# ============================================================
# STEP 3: Train LDA Model
# ============================================================

print("\nTraining LDA model...")

n_topics = 10

lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42
)

lda_model.fit(count_matrix)

print("LDA training complete.")


# ============================================================
# STEP 4: Train NMF Model
# ============================================================

print("\nTraining NMF model...")

nmf_model = NMF(
    n_components=n_topics,
    random_state=42
)

nmf_model.fit(tfidf_matrix)

print("NMF training complete.")


# ============================================================
# STEP 5: Compute Perplexity (LDA)
# ============================================================

lda_perplexity = lda_model.perplexity(count_matrix)

print("\nLDA Perplexity:", round(lda_perplexity, 2))


# ============================================================
# STEP 6: Extract Topics
# ============================================================

def extract_topics(model, feature_names, n_words=10):

    topic_list = []

    for topic in model.components_:

        words = [feature_names[i]
                 for i in topic.argsort()[:-n_words - 1:-1]]

        topic_list.append(words)

    return topic_list


lda_topics = extract_topics(
    lda_model,
    count_vectorizer.get_feature_names_out()
)

nmf_topics = extract_topics(
    nmf_model,
    tfidf_vectorizer.get_feature_names_out()
)


# Convert to pandas DataFrame
lda_df = pd.DataFrame(lda_topics)
nmf_df = pd.DataFrame(nmf_topics)


print("\nLDA Topics:")
print(lda_df)

print("\nNMF Topics:")
print(nmf_df)


# ============================================================
# STEP 7: Save Topics to CSV
# ============================================================

lda_df.to_csv("lda_topics.csv", index=False)
nmf_df.to_csv("nmf_topics.csv", index=False)

print("\nTopics saved to CSV files.")


# ============================================================
# STEP 8: Topic Distribution Plot (Matplotlib)
# ============================================================

print("\nCreating topic distribution plot...")

lda_doc_topics = lda_model.transform(count_matrix)

lda_strength = np.sum(lda_doc_topics, axis=0)

plt.figure(figsize=(10, 5))

plt.bar(range(n_topics), lda_strength)

plt.title("LDA Topic Distribution")

plt.xlabel("Topic Number")

plt.ylabel("Topic Strength")

plt.grid()

plt.show()


# ============================================================
# STEP 9: Compare LDA vs NMF
# ============================================================

print("Creating model comparison plot...")

nmf_doc_topics = nmf_model.transform(tfidf_matrix)

nmf_strength = np.sum(nmf_doc_topics, axis=0)

x = np.arange(n_topics)

width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(x - width/2, lda_strength, width, label="LDA")

plt.bar(x + width/2, nmf_strength, width, label="NMF")

plt.title("LDA vs NMF Topic Strength Comparison")

plt.xlabel("Topic Number")

plt.ylabel("Strength")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# STEP 10: pyLDAvis Dashboard
# ============================================================

print("\nCreating pyLDAvis dashboard...")

lda_vis = pyLDAvis.sklearn.prepare(
    lda_model,
    count_matrix,
    count_vectorizer
)

pyLDAvis.save_html(lda_vis, "lda_dashboard.html")

print("pyLDAvis dashboard saved as lda_dashboard.html")


# ============================================================
# COMPLETED
# ============================================================

print("\n======================================")
print("TASK COMPLETED SUCCESSFULLY")
print("Files generated:")
print("lda_topics.csv")
print("nmf_topics.csv")
print("lda_dashboard.html")
print("======================================")
