from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "This is a simple example showing the removal of stopwords."
tokens = word_tokenize(text)

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print("Before:", tokens)
print("After:", filtered_tokens)