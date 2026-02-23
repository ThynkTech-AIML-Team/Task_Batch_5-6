import nltk
import pandas as pd
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Download and Prepare the Data
nltk.download('movie_reviews')

reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
labels = [movie_reviews.categories(fileid)[0] for fileid in movie_reviews.fileids()]

# 2. Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# 3. Vectorization (Converting text to numbers)
# Added ngram_range to capture phrases like "not good" instead of just "good"
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# 4. Training the Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# 5. Evaluate Accuracy
predictions = model.predict(X_test_vectors)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 6. Predict on your own review
def predict_my_review(text):
    vec = vectorizer.transform([text])
    sentiment = model.predict(vec)[0]
    print(f"\nReview: {text}")
    print(f"Predicted Sentiment: {sentiment.upper()}")

predict_my_review("The plot was predictable and the acting was wooden.")
predict_my_review("A visual masterpiece that redefines the genre!")