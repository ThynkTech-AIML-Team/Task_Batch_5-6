
import nltk
import random
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


nltk.download('twitter_samples')


positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

documents = []


for tweet in positive_tweets:
    documents.append((tweet, "positive"))


for tweet in negative_tweets:
    documents.append((tweet, "negative"))


random.shuffle(documents)


texts = [doc[0] for doc in documents]
labels = [doc[1] for doc in documents]


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)


X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print(twitter_samples.fileids())

# Test 
new_tweet = ["This phone is amazing and I love it"]
new_tweet_vector = vectorizer.transform(new_tweet)
prediction = model.predict(new_tweet_vector)

print("\nTweet Prediction:", prediction[0])