import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "AI is transforming the future of warfare and strategy",
    "Artificial intelligence enhances predictive military systems",
    "Strategy and intelligence define the outcome of war",
    "Future systems rely on AI and data driven intelligence"
]

# 1
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(sentences)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_vectorizer.get_feature_names_out()
)

print("\n=== Bag of Words Matrix ===\n")
print(bow_df)

#2
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\n=== TF-IDF Matrix ===\n")
print(tfidf_df.round(3))

# 3
bow_word_importance = bow_df.sum().sort_values(ascending=False)
tfidf_word_importance = tfidf_df.sum().sort_values(ascending=False)

comparison = pd.DataFrame({
    "BoW_Total_Count": bow_word_importance,
    "TFIDF_Total_Score": tfidf_word_importance
}).sort_values("TFIDF_Total_Score", ascending=False)

print("\n=== Global Word Importance Comparison ===\n")
print(comparison.round(3))

# 4
print("\n=== Top 3 Important Words Per Sentence (TF-IDF) ===\n")

feature_names = tfidf_vectorizer.get_feature_names_out()

for i, row in enumerate(tfidf_df.values):
    top_indices = row.argsort()[-3:][::-1]
    top_words = [(feature_names[j], row[j]) for j in top_indices]
    print(f"Sentence {i+1}: {top_words}")

# 5
bow_similarity = cosine_similarity(bow_matrix)
tfidf_similarity = cosine_similarity(tfidf_matrix)

print("\n=== Cosine Similarity (BoW) ===\n")
print(pd.DataFrame(bow_similarity).round(3))

print("\n=== Cosine Similarity (TF-IDF) ===\n")
print(pd.DataFrame(tfidf_similarity).round(3))