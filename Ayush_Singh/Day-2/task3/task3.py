# ============================================================
# TASK 3: End-to-End NLP Application
# Application: News Similarity Search
# Input: One news article
# Output: Top 3 most similar articles
# ============================================================

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# STEP 1: Load Dataset
# ============================================================

print("Loading dataset...")

dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

documents = dataset.data[:2000]

df = pd.DataFrame({

    "Article": documents

})

print("Dataset loaded successfully.")

print("Total articles:", len(df))


# ============================================================
# STEP 2: Convert Articles to TF-IDF Vectors
# ============================================================

print("\nCreating TF-IDF vectors...")

vectorizer = TfidfVectorizer(

    stop_words="english",

    max_df=0.95,

    min_df=2

)

tfidf_matrix = vectorizer.fit_transform(df["Article"])

print("TF-IDF matrix shape:", tfidf_matrix.shape)


# ============================================================
# STEP 3: Similarity Function
# ============================================================

def find_similar_articles(input_article, top_n=3):

    # Convert input article to vector
    input_vector = vectorizer.transform([input_article])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)

    similarity_scores = similarity_scores.flatten()

    # Get top similar indices
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    results = []

    for index in top_indices:

        results.append({

            "Article Index": index,

            "Similarity Score": similarity_scores[index],

            "Article Text": df.iloc[index]["Article"][:300]

        })

    return pd.DataFrame(results)


# ============================================================
# STEP 4: User Input
# ============================================================

print("\nEnter a news article:")

user_article = input()


# ============================================================
# STEP 5: Find Similar Articles
# ============================================================

results_df = find_similar_articles(user_article, top_n=3)


print("\nTop 3 Similar Articles:\n")

print(results_df)


# ============================================================
# STEP 6: Save Results
# ============================================================

results_df.to_csv("similar_articles.csv", index=False)

print("\nResults saved to similar_articles.csv")


# ============================================================
# STEP 7: Example Automatic Test
# ============================================================

example_article = documents[0]

example_results = find_similar_articles(example_article)

example_results.to_csv("example_results.csv", index=False)


# ============================================================
# COMPLETED
# ============================================================

print("\n===================================")
print("TASK 3 COMPLETED SUCCESSFULLY")
print("Generated files:")
print("similar_articles.csv")
print("example_results.csv")
print("===================================")
