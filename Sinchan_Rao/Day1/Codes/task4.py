from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

sentences = [
    "NLP is fun and interesting",
    "NLP is powerful and useful",
    "Machine learning is powerful"
]

bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(sentences)

print("===== Bag of Words Matrix =====")
print(pd.DataFrame(bow_matrix.toarray(),
                   columns=bow_vectorizer.get_feature_names_out()))

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

print("\n===== TF-IDF Matrix =====")
print(pd.DataFrame(tfidf_matrix.toarray(),
                   columns=tfidf_vectorizer.get_feature_names_out()))

print("\n===== Word Importance Comparison =====")

bow_importance = bow_matrix.toarray().sum(axis=0)
tfidf_importance = tfidf_matrix.toarray().sum(axis=0)

comparison = pd.DataFrame({
    "Word": bow_vectorizer.get_feature_names_out(),
    "BoW Score (Frequency)": bow_importance,
    "TF-IDF Score (Importance)": tfidf_importance
})

print(comparison.sort_values(by="TF-IDF Score (Importance)", ascending=False))
