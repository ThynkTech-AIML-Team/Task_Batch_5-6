# Task 4: Bag-of-Words (BoW) and TF-IDF

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd


sentences = [
    "I love natural language processing",
    "Natural language processing is powerful",
    "I love machine learning",
    "Machine learning and NLP are related"
]

bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(sentences)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_vectorizer.get_feature_names_out()
)

print("=== BAG-OF-WORDS MATRIX ===")
print(bow_df)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\n=== TF-IDF MATRIX ===")
print(tfidf_df)

print("\n=== WORD IMPORTANCE COMPARISON ===")

for word in tfidf_vectorizer.get_feature_names_out():
    bow_score = bow_df[word].sum()
    tfidf_score = tfidf_df[word].sum()
    print(f"{word} -> BoW: {bow_score}, TF-IDF: {tfidf_score:.3f}")
