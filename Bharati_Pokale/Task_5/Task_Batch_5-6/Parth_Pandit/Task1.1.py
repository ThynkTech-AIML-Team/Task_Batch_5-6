# Task 1: Introduction to NLP

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

with open("sentiment_data.txt", "r", encoding="utf-8") as file:
    texts = [line.strip() for line in file if line.strip()]


labels = [1] * (len(texts) // 2) + [0] * (len(texts) // 2)

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

new_texts = [
    "The movie was absolutely fantastic",
    "I did not like this film at all",
    "The acting was boring and poor",
    "Amazing story and great direction"
]

new_vectors = vectorizer.transform(new_texts)
predictions = model.predict(new_vectors)

print("Sentiment Analysis Results:\n")

for text, pred in zip(new_texts, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print("-" * 40)
