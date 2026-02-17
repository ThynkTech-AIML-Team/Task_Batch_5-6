
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sentences = [
    "NLP is fun",
    "I love NLP",
    "NLP is powerful and fun"
]

print("Original Sentences:")
for s in sentences:
    print(s)


print("\n--- Bag of Words Matrix ---")

bow_vectorizer = CountVectorizer()

bow_matrix = bow_vectorizer.fit_transform(sentences)

bow_array = bow_matrix.toarray()

print("Words:", bow_vectorizer.get_feature_names_out())
print("BoW Matrix:")
print(bow_array)

print("\n--- TF-IDF Matrix ---")

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

tfidf_array = tfidf_matrix.toarray()

print("Words:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix:")
print(tfidf_array)

print("\n--- Word Importance Comparison ---")

words = tfidf_vectorizer.get_feature_names_out()

for i, sentence in enumerate(sentences):
    print("\nSentence:", sentence)
    for j, word in enumerate(words):
        print(word, "=", round(tfidf_array[i][j], 2))
