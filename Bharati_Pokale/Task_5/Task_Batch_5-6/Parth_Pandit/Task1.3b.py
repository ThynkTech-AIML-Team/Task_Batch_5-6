# Task 3(B): Stopwords Removal

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

text = "This is an example sentence showing how stopwords are removed from a text"

tokens = word_tokenize(text)

stop_words = set(stopwords.words('english'))

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print("=== 3(B) STOPWORDS REMOVAL ===")
print("Original Text:")
print(text)

print("\nTokens Before Stopword Removal:")
print(tokens)

print("\nTokens After Stopword Removal:")
print(filtered_tokens)
