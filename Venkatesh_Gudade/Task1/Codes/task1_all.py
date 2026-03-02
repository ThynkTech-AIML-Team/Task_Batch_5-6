# =========================================
# TASK 1 — NLP APPLICATIONS (FULL CLEAN)
# Part A — Sentiment Analysis
# Part B — Text Summarization
# =========================================

import nltk

# ---- First-time downloads (safe to keep) ----
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('punkt_tab')

# =========================================
# TASK 1A — SENTIMENT ANALYSIS
# =========================================

print("\n===== TASK 1A — SENTIMENT ANALYSIS =====")

from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load movie review dataset
documents = [" ".join(movie_reviews.words(fid)) for fid in movie_reviews.fileids()]
labels = [movie_reviews.categories(fid)[0] for fid in movie_reviews.fileids()]

# Better vectorizer for sentiment
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(documents)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", round(accuracy, 3))

# Test samples
samples = [
    "The movie was absolutely wonderful and inspiring",
    "This was the worst movie I have ever seen"
]

print("\nPredictions:")
for text in samples:
    pred = model.predict(vectorizer.transform([text]))[0]
    print("Text:", text)
    print("Predicted Sentiment:", pred.upper())
    print()

# =========================================
# TASK 1B — TEXT SUMMARIZATION
# =========================================

print("\n===== TASK 1B — TEXT SUMMARIZATION =====")

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

text = """
Natural language processing helps computers understand human language.
It is used in chatbots, translation systems, and sentiment analysis.
NLP is an important part of modern artificial intelligence.
Many companies use NLP in real-world products.
AI systems rely heavily on language understanding.
"""

parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = LsaSummarizer()

summary = summarizer(parser.document, 2)

print("\nSummary:")
for sentence in summary:
    print(sentence)
