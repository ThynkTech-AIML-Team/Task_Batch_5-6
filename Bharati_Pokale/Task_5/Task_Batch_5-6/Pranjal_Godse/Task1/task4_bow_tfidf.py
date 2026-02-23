from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sentences = [
    "NLP is fun",
    "NLP is powerful",
    "Machine learning is fun"
]

# Bag of Words
cv = CountVectorizer()
bow = cv.fit_transform(sentences)

print("BoW Matrix:")
print(bow.toarray())
print("Feature Names:", cv.get_feature_names_out())

# using TF-IDF here
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(sentences)

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("Feature Names:", tfidf.get_feature_names_out())
