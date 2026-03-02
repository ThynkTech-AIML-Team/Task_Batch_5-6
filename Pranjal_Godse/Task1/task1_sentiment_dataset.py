import nltk
import random
from nltk.corpus import movie_reviews
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download dataset and sentiment resources
nltk.download('movie_reviews')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

# Load dataset
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

texts = [doc for doc, label in documents]
labels = [label for doc, label in documents]

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
#print(movie_reviews.fileids())


# SENTIMENT ANALYSIS (POSITIVE/NEGATIVE)

print("\n\n" + "="*50)
print("SENTIMENT ANALYSIS (VADER)")
print("="*50)

sia = SentimentIntensityAnalyzer()

# Test sentiment analysis on sample reviews
sample_reviews = [
    "This movie was absolutely amazing! I loved every minute of it.",
    "Terrible film. Complete waste of time. Very disappointed.",
    "It was okay, nothing special but watchable.",
    "Best movie I've ever seen! Outstanding performance!",
    "Horrible acting and poor storyline. Don't recommend."
]

print("\nSentiment Analysis Results:")
print("-" * 50)

for review in sample_reviews:
    scores = sia.polarity_scores(review)
    sentiment = "POSITIVE" if scores['compound'] >= 0.05 else "NEGATIVE" if scores['compound'] <= -0.05 else "NEUTRAL"
    print(f"\nReview: {review[:60]}...")
    print(f"Scores: {scores}")
    print(f"Sentiment: {sentiment}")

# ============================================
# TEXT SUMMARIZATION (EXTRACTIVE - NLTK BASED)
# ============================================
print("\n\n" + "="*50)
print("TEXT SUMMARIZATION (EXTRACTIVE)")
print("="*50)

def summarize_text(text, num_sentences=3):
    """Simple extractive summarization using TF-IDF"""
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return sentences
    
    # Create a simple TF-IDF vectorizer for sentences
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Sum scores for each sentence
    sentence_scores = tfidf_matrix.sum(axis=1).A1
    
    # Get top sentences
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    top_indices = sorted(top_indices)
    
    return [sentences[i] for i in top_indices]

# Test summarization
sample_text = """
This movie was an absolute masterpiece. The director's vision was perfectly executed, 
with stunning cinematography that captured every emotional moment. The cast delivered 
outstanding performances, particularly the lead actor whose portrayal was nuanced and 
compelling. The soundtrack perfectly complemented the scenes, enhancing emotional depth. 
The plot was engaging from start to finish, with surprising twists that kept audiences 
on the edge of their seats. The dialogue felt natural and authentic. Special effects were 
seamlessly integrated without overshadowing the human performances. The pacing was excellent, 
never feeling rushed or dragging. Overall, this is a film that will be remembered as a classic.
"""

print("\nOriginal Text Length:", len(sample_text.split()), "words")
print("\nOriginal Text:")
print(sample_text[:200] + "...")

# Summarize
summary_sentences = summarize_text(sample_text, num_sentences=3)

print("\n\nSummary (3 sentences):")
print("-" * 50)
for i, sentence in enumerate(summary_sentences, 1):
    print(f"{i}. {sentence}")

print("\nSummary Length:", sum(len(s.split()) for s in summary_sentences), "words")
