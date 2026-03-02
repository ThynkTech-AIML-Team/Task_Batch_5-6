import nltk
import re
import numpy as np
from nltk.corpus import movie_reviews, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# downloading data
nltk.download('movie_reviews', quiet=True)
nltk.download('stopwords', quiet=True)

print("\n=== Task 1: Sentiment Analysis ===")
print("Checking if movie reviews are positive or negative\n")

# load movie reviews
print("loading data...")
reviews = []
labels = []
for fileid in movie_reviews.fileids():
    reviews.append(movie_reviews.raw(fileid))
    labels.append(1 if movie_reviews.categories(fileid)[0] == 'pos' else 0)  # 1=positive, 0=negative

print(f"got {len(reviews)} reviews")
print(f"positive: {sum(labels)}, negative: {len(labels) - sum(labels)}\n")

# basic preprocessing - lowercase and remove punctuation
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    return text

print("cleaning up reviews...")
reviews_clean = [preprocess(r) for r in reviews]

# converting to features
print("converting to TF-IDF")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(reviews_clean)
y = np.array(labels)
print(f"feature matrix size: {X.shape}\n")

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"training on {X_train.shape[0]} reviews, testing on {X_test.shape[0]}\n")

# train the model
print("training...")
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# check how well it did
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\naccuracy: {accuracy:.2%}")
print(f"so it gets about {accuracy:.0%} of them right")
