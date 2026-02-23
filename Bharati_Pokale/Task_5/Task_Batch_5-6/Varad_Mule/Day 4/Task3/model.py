import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("titanic.csv")

df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], errors="ignore")

if "Sex" in df.columns and df["Sex"].dtype == "object":
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df = df.fillna(df.median(numeric_only=True))

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
X = df.drop("Survived", axis=1)
y = df["Survived"]
X = X.select_dtypes(include=["number"])

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "titanic_model.pkl")
print("Model trained and saved successfully!")