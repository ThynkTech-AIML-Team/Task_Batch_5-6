import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("titanic.csv")

# Fill missing values
df["age"].fillna(df["age"].median(), inplace=True)

# Feature Engineering
df["familysize"] = df["sibsp"] + df["parch"] + 1
df["isalone"] = df["familysize"].apply(lambda x: 0 if x > 1 else 1)

# Encode sex
df["sex"] = df["sex"].map({"male": 0, "female": 1})

# Drop name column (not useful for model)
df.drop("name", axis=1, inplace=True)

# Define features and target
X = df.drop("survived", axis=1)
y = df["survived"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
pickle.dump(model, open("titanic_model.pkl", "wb"))

print("Model trained and saved successfully!")