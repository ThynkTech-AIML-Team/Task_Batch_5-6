from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

#input sentences
sentences = [
    "The cat sat on the mat.",
    "The dog chased the cat.",
    "The dog and the cat are friends."
]

#Bag-of-Words (BoW)
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(sentences)

#Convert to DataFrame for visualization
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
print("Bag-of-Words Matrix:\n", bow_df)

#TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

#Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:\n", tfidf_df)

#Compare word importance
comparison_df = pd.DataFrame({
    'word': bow_vectorizer.get_feature_names_out(),
    'BoW_count': bow_df.sum(axis=0).values,
    'TF-IDF_avg': tfidf_df.mean(axis=0).values
})
print("\nWord Importance Comparison (BoW count vs TF-IDF average):")
print(comparison_df.sort_values(by='TF-IDF_avg', ascending=False))
