import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('movie_reviews')

reviews = []
labels = []

for fileid in movie_reviews.fileids():
    reviews.append(movie_reviews.raw(fileid))
    labels.append(movie_reviews.categories(fileid)[0])

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

model = MultinomialNB()
model.fit(X, labels)

# Multiple
test_reviews = [
    "This movie is very bad.",
    "I really loved this film.",
    "It was an average movie."
]

test_vectors = vectorizer.transform(test_reviews)
predictions = model.predict(test_vectors)

for review, sentiment in zip(test_reviews, predictions):
    print("Review:", review)
    print("Sentiment:", sentiment)
    print("-" * 40)
