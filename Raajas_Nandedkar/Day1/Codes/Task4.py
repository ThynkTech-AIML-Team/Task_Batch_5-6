from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Sample sentences (you can change these)
sentences = [
    "NLP is fun and useful",
    "NLP is powerful",
    "I love learning NLP"
]

print("Sentences:")
for s in sentences:
    print("-", s)

# -------------------------
# Bag of Words
# -------------------------
bow = CountVectorizer()
bow_matrix = bow.fit_transform(sentences)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow.get_feature_names_out()
)

print("\n=== Bag of Words Matrix ===")
print(bow_df)

# -------------------------
# TF-IDF
# -------------------------
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(sentences)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

print("\n=== TF-IDF Matrix ===")
print(tfidf_df.round(3))


#-----
print("\n=== Word Importance Example ===")
word = "nlp"
if word in tfidf_df.columns:
    print(f"TF-IDF scores for '{word}':")
    print(tfidf_df[word])
else:
    print("Word not found in vocabulary")