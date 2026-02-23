# 🏦 Loan Approval Prediction System

This project implements a Machine Learning system to predict whether a loan application will be approved or rejected based on various applicant details.

## 📋 Project Overview
The goal is to automate the loan eligibility process based on customer details provided during the online application. These details include Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others.

## 🛠️ Tech Stack
- **Python 3.11.1**
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Environment**: Virtual Environment (venv)
- **Interactive Tool**: Jupyter Notebook

## 📊 Model Comparison
The following models were trained and evaluated on the dataset:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 78.86% |
| Decision Tree | 69.11% |
| Random Forest | 75.61% |

*Note: Results may vary slightly depending on data splitting and random seeds.*

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.11.1 installed.

### 2. Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Running the Project
Open the Jupyter Notebook to see the step-by-step analysis:
```bash
jupyter notebook Loan_Approval_Prediction.ipynb
```

## 📂 Project Structure
- `Loan_Approval_Prediction.ipynb`: Main analysis notebook.
- `loan_data.csv`: Dataset used for training.
- `requirements.txt`: List of dependencies.
- `venv/`: Virtual environment folder.
