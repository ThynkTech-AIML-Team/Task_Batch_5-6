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

test_review = ["This movie is very good and amazing"]
test_vector = vectorizer.transform(test_review)

prediction = model.predict(test_vector)

print("Review:", test_review[0])
print("Sentiment:", prediction[0])
