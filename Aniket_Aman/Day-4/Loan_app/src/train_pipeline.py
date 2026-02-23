import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_real_data():
    """Loads the standard real-world Loan Prediction dataset."""
    print("Fetching real-world dataset...")
    url = "https://raw.githubusercontent.com/sahutkarsh/loan-prediction-analytics-vidhya/master/train.csv"
    df = pd.read_csv(url)
    
    # Drop Loan_ID as it has no predictive value
    df = df.drop('Loan_ID', axis=1)
    
    # Map the target variable to binary (1 = Approved, 0 = Rejected)
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Drop any rows where the target itself is missing
    df = df.dropna(subset=['Loan_Status'])
    return df

def main():
    df = load_real_data()
    
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Define Numeric and Categorical Features
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    # 2. Build Transformers with Imputation for missing values
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Fills missing numbers with the median
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Fills missing categories with the mode
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 3. Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # 4. Build the final classification pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42))
    ])

    print("Training the classification pipeline on real data...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate the Model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.2%}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 6. Save the Pipeline
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/real_loan_pipeline.pkl'
    joblib.dump(pipeline, model_path)
    print(f"Pipeline saved successfully to {model_path}!")

if __name__ == "__main__":
    main()