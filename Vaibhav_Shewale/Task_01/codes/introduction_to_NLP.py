import pandas as pd
import re
import joblib
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------
# Text Cleaning
# -------------------------------
def clean_text(text):
    text = re.sub('<[^<]+?>', '', str(text))
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# Load Data
# -------------------------------
logging.info("Loading dataset...")
df = pd.read_csv("IMDB Dataset.csv")

df["review"] = df["review"].apply(clean_text)

X = df["review"]
y = df["sentiment"]

# -------------------------------
# Train-Test Split (Stratified)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# ML Pipeline
# -------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=2000))
])

# -------------------------------
# Hyperparameter Tuning
# -------------------------------
param_grid = {
    "tfidf__max_features": [5000, 10000],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__C": [0.1, 1, 10]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

logging.info("Training model with GridSearch...")
grid.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

logging.info(f"Best Parameters: {grid.best_params_}")
logging.info(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print(classification_report(y_test, y_pred))

# -------------------------------
# Save Model
# -------------------------------
joblib.dump(best_model, "sentiment_model.pkl")
logging.info("Model saved as sentiment_model.pkl")