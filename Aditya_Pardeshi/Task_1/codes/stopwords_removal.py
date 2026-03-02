import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Sample text
text = "Artificial Intelligence is transforming the world and it is making life easier."

# Tokenize text
tokens = word_tokenize(text)

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Print results
print("Original Text:\n", text)

print("\nBefore Stopword Removal:")
print(tokens)

print("\nAfter Stopword Removal:")
print(filtered_tokens)
