# Task 3(C): Lemmatization & Stemming

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')

words = ["studies", "studying", "running", "better", "cars"]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("=== 3(C) STEMMING vs LEMMATIZATION ===")
print("Word | Stemmed | Lemmatized")
print("-" * 35)

for word in words:
    stemmed = stemmer.stem(word)
    lemmatized = lemmatizer.lemmatize(word)
    print(f"{word} | {stemmed} | {lemmatized}")
