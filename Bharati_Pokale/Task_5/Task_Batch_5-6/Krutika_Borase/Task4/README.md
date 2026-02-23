# Titanic Survival Prediction Project

## Overview
This project analyzes the Titanic dataset, performs exploratory data analysis (EDA), builds and compares several machine learning models, and deploys a web app for survival prediction.

## Project Structure
- `1_titanic_eda.ipynb`: Exploratory Data Analysis
- `2_titanic_ml.ipynb`: Machine Learning Model Development & Comparison
- `3_app.py`: Streamlit Web App for Prediction
- `Titanic-Dataset.csv`: Dataset

## Model Comparison Table
Model Performance Comparison:
======================================================================
              Model  Before Tuning  After Tuning  Improvement
Logistic Regression       0.804196      0.797203    -0.006993
      Decision Tree       0.783217      0.804196     0.020979
      Random Forest       0.783217      0.797203     0.013986
                SVM       0.825175      0.825175     0.000000
  Gradient Boosting       0.797203      0.811189     0.013986
            XGBoost       0.762238      0.804196     0.041958
======================================================================

**Dataset:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
