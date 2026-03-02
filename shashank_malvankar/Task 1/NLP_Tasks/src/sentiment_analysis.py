import os
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

DATA_PATH = os.path.join("data", "sample_reviews.txt")
OUTPUT_PATH = os.path.join("outputs", "sentiment_results.txt")

def load_data(path):
    with open(path, "r", encoding="utf-8") as file:
        reviews = file.readlines()

    reviews = [review.strip() for review in reviews if review.strip()]
    return reviews

def create_labels(reviews):
    labels = []

    positive_words = ["amazing", "fantastic", "loved", "enjoyed"]
    negative_words = ["terrible", "worst", "boring", "dull"]

    for review in reviews:
        if any(word in review.lower() for word in positive_words):
            labels.append(1)
        else:
            labels.append(0)

    return labels

def train_model(reviews, labels):
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(reviews)
    model = LogisticRegression()

    model.fit(X, labels)

    return model, vectorizer

def predict_sentiment(model, vectorizer, reviews):
    X = vectorizer.transform(reviews)
    predictions = model.predict(X)

    return predictions

def save_results(reviews, predictions, path):

    os.makedirs("outputs", exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:

        file.write("Sentiment Analysis Results\n")
        file.write("=========================\n\n")

        for review, prediction in zip(reviews, predictions):

            sentiment = "Positive" if prediction == 1 else "Negative"

            file.write(f"Review: {review}\n")
            file.write(f"Sentiment: {sentiment}\n")
            file.write("-------------------------\n")

def main():

    print("Loading data...")
    reviews = load_data(DATA_PATH)

    print("Creating labels...")
    labels = create_labels(reviews)

    print("Training model...")
    model, vectorizer = train_model(reviews, labels)

    print("Predicting sentiment...")
    predictions = predict_sentiment(model, vectorizer, reviews)

    print("Saving results...")
    save_results(reviews, predictions, OUTPUT_PATH)

    print("Done. Results saved in outputs/sentiment_results.txt")

if __name__ == "__main__":
    main()
