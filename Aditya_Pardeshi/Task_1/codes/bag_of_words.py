from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Sample data
sentences = [
    "I love NLP",
    "I love machine learning",
    "NLP is powerful and learning is fun"
]

# ----------- 1️⃣ Bag of Words -----------
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(sentences)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_vectorizer.get_feature_names_out()
)

print("Bag of Words Matrix:\n")
print(bow_df)


# ----------- 2️⃣ TF-IDF -----------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\nTF-IDF Matrix:\n")
print(tfidf_df)
