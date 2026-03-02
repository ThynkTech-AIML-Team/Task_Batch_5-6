# =====================================================
# DAY 3 - TASK 1: SMS SPAM CLASSIFICATION SYSTEM
# Author: Ayush Singh
# =====================================================

print("\n==========================================")
print("SMS SPAM DETECTION SYSTEM")
print("==========================================")

# Imports
import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords

import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')

# =====================================================
# STEP 1: LOAD DATASET
# =====================================================

print("\nLoading dataset...")

df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1','v2']]
df.columns = ['label','text']

print("Dataset loaded successfully")
print("Total messages:", len(df))

# Convert labels
df['label'] = df['label'].map({'ham':0,'spam':1})

# =====================================================
# STEP 2: CLEAN TEXT
# =====================================================

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = text.lower()

    text = re.sub(f"[{string.punctuation}]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

print("\nCleaning text...")

df['clean_text'] = df['text'].apply(clean_text)

print("Cleaning completed")

# =====================================================
# STEP 3: TRAIN TEST SPLIT
# =====================================================

X = df['clean_text']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

print("\nTraining samples:",len(X_train))
print("Testing samples:",len(X_test))

# =====================================================
# STEP 4: TF-IDF
# =====================================================

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nFeature size:",X_train_vec.shape)

# =====================================================
# STEP 5: TRAIN MODELS
# =====================================================

models = {

    "Naive Bayes":MultinomialNB(),

    "Logistic Regression":LogisticRegression(max_iter=1000),

    "SVM":SVC()
}

results = {}

for name,model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_vec,y_train)

    pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test,pred)

    results[name]=acc

    print("Accuracy:",acc)

    print("\nClassification Report:")
    print(classification_report(y_test,pred))

    cm = confusion_matrix(y_test,pred)

    print("Confusion Matrix:")
    print(cm)

    plt.figure()
    sns.heatmap(cm,annot=True,fmt='d')
    plt.title(name)
    plt.show()

# =====================================================
# STEP 6: RESULTS
# =====================================================

print("\nFINAL RESULTS")

for model,acc in results.items():
    print(model,":",acc)

# =====================================================
# STEP 7: INTERACTIVE MODE
# =====================================================

best_model = models["Logistic Regression"]

print("\n====================================")
print("INTERACTIVE SPAM DETECTOR")
print("====================================")

while True:

    msg=input("\nEnter message (exit to quit): ")

    if msg=="exit":
        break

    msg_clean=clean_text(msg)

    vec=vectorizer.transform([msg_clean])

    pred=best_model.predict(vec)[0]

    if pred==1:
        print("SPAM")
    else:
        print("HAM")
