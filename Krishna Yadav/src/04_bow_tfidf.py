from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

print(">>> Script started")

# ------------------------------------------
# Example Sentences
# ------------------------------------------
sentences = [
    "NLP is fun",
    "NLP is powerful",
    "Machine learning is fun"
]

print("\nOriginal Sentences:")
print(sentences)


# ------------------------------------------
# Bag of Words (BoW)
# ------------------------------------------
print("\n========== BAG OF WORDS ==========\n")

bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(sentences)

print("Vocabulary:", bow_vectorizer.get_feature_names_out())
print("\nBoW Matrix:")
print(X_bow.toarray())


# ------------------------------------------
# TF-IDF
# ------------------------------------------
print("\n========== TF-IDF ==========\n")

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(sentences)

print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:")
print(X_tfidf.toarray())


print("\n>>> Script finished")
