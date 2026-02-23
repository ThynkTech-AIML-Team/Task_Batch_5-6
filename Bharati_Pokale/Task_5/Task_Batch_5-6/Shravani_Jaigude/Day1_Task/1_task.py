import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/SMSSpamCollection"

df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)      # remove punctuation + numbers
    text = re.sub(r"\s+", " ", text).strip() # normalize spaces
    return text

df["clean_message"] = df["message"].apply(clean_text)


df["label_num"] = df["label"].map({"ham": 0, "spam": 1})


X_train, X_test, y_train, y_test = train_test_split(
    df["clean_message"],
    df["label_num"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_num"]
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("\n--- MODEL RESULTS ---")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])
    pred = model.predict(msg_vec)[0]
    return "SPAM!!" if pred == 1 else "HAM:)"

print("\n--- CUSTOM TESTS ---")
print("1)", predict_message("Congratulations! You won a free iPhone. Click now!"))
print("2)", predict_message("Hey Shravani, are you coming to college today?"))
print("3)", predict_message("URGENT! You have won cash prize. Call now!!!"))
