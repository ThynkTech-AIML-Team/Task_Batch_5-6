from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

corpus = [
    "I love programming in Python.",
    "Python is great for Data Science.",
    "I love Data Science."
]

# --- Bag of Words ---
print("\n--- Task 4: Bag of Words (BoW) ---")
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(corpus)
df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer_bow.get_feature_names_out())
print(df_bow)

# --- TF-IDF ---
print("\n--- Task 4: TF-IDF ---")
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(corpus)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out())
print(df_tfidf)

# Comparing importance
print("\nWord Importance (TF-IDF vs BoW for Sentence 2):")
print(f"Sentence: '{corpus[1]}'")
word = "python"
print(f"BoW Count for '{word}': {df_bow.loc[1, word]}")
print(f"TF-IDF Score for '{word}': {df_tfidf.loc[1, word]:.4f}")