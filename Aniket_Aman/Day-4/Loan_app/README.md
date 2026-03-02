# 🏦 Nexus Bank: Automated Loan Approval System

## 📌 Project Overview
This project is an end-to-end Machine Learning deployment that predicts whether a bank loan should be approved or rejected based on applicant profiling. It features a robust Scikit-Learn data processing pipeline and an interactive web dashboard built with Streamlit.

This project was built to simulate a real-world enterprise underwriting system, focusing on data imputation, pipeline architecture, and user-friendly deployment.

## 🎯 Objective
To build a production-ready classification model using real-world financial data, encapsulating data preprocessing and model inference within a unified pipeline, and deploying it via a web interface.

## 📊 Dataset
* **Source:** Analytics Vidhya Loan Prediction Dataset.
* **Description:** A real-world dataset containing historical loan application data, including applicant demographics, income, requested loan amounts, and credit history. It includes real-world noise and missing values (NaNs).

## 🛠️ Tech Stack & Architecture
* **Frontend/Deployment:** Streamlit
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`)
* **Data Manipulation:** Pandas, NumPy
* **Model Serialization:** Joblib

## ⚙️ The Machine Learning Pipeline
Instead of manually cleaning data before prediction, this project utilizes a `scikit-learn` `Pipeline` to prevent data leakage and ensure seamless inference in production. 
* **Numeric Features:** Handled via `SimpleImputer` (median strategy) and `StandardScaler`.
* **Categorical Features:** Handled via `SimpleImputer` (most frequent strategy) and `OneHotEncoder`.
* **Classifier:** A Random Forest model optimized with balanced class weights to handle unequal approval/rejection ratios in the historical data.

## 📁 Project Structure
```text
Task4_LoanApp/
│
├── models/
│   └── real_loan_pipeline.pkl   # Serialized ML pipeline (generated after training)
├── src/
│   └── train_pipeline.py        # Fetches data, builds pipeline, trains, and saves model
├── app.py                       # Streamlit web application
└── README.md                    # Project documentation