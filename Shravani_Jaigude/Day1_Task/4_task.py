from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import os


sentences = [
    "I love NLP",
    "NLP is fun",
    "I love machine learning"
]

os.makedirs("outputs", exist_ok=True)

bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(sentences)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_vectorizer.get_feature_names_out()
)

print("\n--- BoW Matrix ---")
print(bow_df)

bow_df.to_csv("outputs/bow_matrix.csv", index=False)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\n--- TF-IDF Matrix ---")
print(tfidf_df)

tfidf_df.to_csv("outputs/tfidf_matrix.csv", index=False)
