# train_model.py

import pandas as pd
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = sns.load_dataset("titanic")

# Clean data
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Feature engineering
df["family_size"] = df["sibsp"] + df["parch"] + 1

# Select useful columns
df = df[["survived", "pclass", "sex", "age", "fare", "family_size"]]

# Encode sex
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])

# Split data
X = df.drop("survived", axis=1)
y = df["survived"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully")
