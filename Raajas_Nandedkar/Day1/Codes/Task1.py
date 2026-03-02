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

test_review = ["This movie is very bad"]
test_vector = vectorizer.transform(test_review)

prediction = model.predict(test_vector)

print("Review:", test_review[0])
print("Sentiment:", prediction[0])
##########################################################################
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from CSV
df = pd.read_csv("movie_reviews.csv")

# Separate features and labels
texts = df["review"]
labels = df["sentiment"]

# Convert text to numerical features
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


custominput=["movie is vvery good . i love it"]
customvector=vectorizer.transform(custominput)
prediction=model.predict(customvector)
print(prediction)



from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Sample open text dataset (can replace with news article or file)
text = """
An unmanned aerial vehicle (UAV) is defined as a "powered, aerial vehicle that does not carry a human operator, uses aerodynamic forces to provide vehicle lift, can fly autonomously or be piloted remotely, can be expendable or recoverable, and can carry a lethal or nonlethal payload".[14] UAV is a term that is commonly applied to military use cases.[15] Missiles with warheads are generally not considered UAVs because the vehicle itself is a munition, but certain types of propeller-based missile are often called "kamikaze drones" by the public and media. Also, the relation of UAVs to remote controlled model aircraft is unclear in some jurisdictions. The US FAA now defines any unmanned flying craft as a UAV regardless of weight.[16] Similar terms are remotely piloted aircraft (RPA) and remotely piloted aerial vehicle (RPAV).
"""

# Parse text
parser = PlaintextParser.from_string(text, Tokenizer("english"))

# Create summarizer
summarizer = LsaSummarizer()

# Generate summary (3 sentences)
summary = summarizer(parser.document, 3)

print("Summary:\n")
for sentence in summary:
    print(sentence)