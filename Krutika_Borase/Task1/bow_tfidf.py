import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

print("\n=== Task 4: BoW and TF-IDF ===")
print()

# example sentences
sentences = [
    "NLP is fun and powerful.",
    "NLP enables computers to understand language.",
    "Language processing is a key part of AI."
]

print("sentences:")
for i, sentence in enumerate(sentences, 1):
    print(f"  {i}. {sentence}")
print()

# bag of words - counts how many times each word appears
print("Bag-of-Words:")

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(sentences)

print(f"vocabulary: {list(bow_vectorizer.get_feature_names_out())}\n")

print("matrix:")
print(bow.toarray())
print()

# TF-IDF - gives importance scores
print("TF-IDF:")

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(sentences)

print(f"vocabulary: {list(tfidf_vectorizer.get_feature_names_out())}\n")

print("matrix:")
print(np.round(tfidf.toarray(), 2))
print()

# comparing both
print("Comparing BoW vs TF-IDF:")
print(f"{'word':<15} {'BoW':<10} {'TF-IDF':<10}")

for i, word in enumerate(tfidf_vectorizer.get_feature_names_out()):
    bow_total = bow.toarray()[:, i].sum()
    tfidf_total = tfidf.toarray()[:, i].sum()
    print(f"{word:<15} {bow_total:<10} {tfidf_total:<10.2f}")

print("\nwhat's the difference?")
print("- BoW just counts words")
print("- TF-IDF scores unique words higher")
print("- common words get lower tfidf scores\n")

# most important word per sentence
print("most important word in each:")
for i, sentence in enumerate(sentences):
    scores = tfidf.toarray()[i]
    max_idx = scores.argmax()
    max_word = tfidf_vectorizer.get_feature_names_out()[max_idx]
    print(f"  {i+1}. {max_word} - \"{sentence}\"")
