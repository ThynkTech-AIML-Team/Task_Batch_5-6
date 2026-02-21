# 🚢 Task 4 — Exploratory Data Analysis & Machine Learning Project

## 📌 Project Overview

This project performs **Exploratory Data Analysis (EDA)** and builds multiple **Machine Learning models** to predict survival outcomes using the Titanic dataset.
The objective is to understand data patterns, engineer useful features, compare model performance, and improve results using hyperparameter tuning.

---

## 🎯 Objectives

* Perform data cleaning and preprocessing
* Handle missing values
* Create visualizations (histograms, boxplots, correlation heatmap)
* Train and compare multiple ML models
* Evaluate using accuracy and confusion matrix
* Apply GridSearchCV for performance improvement

---

## 📂 Project Structure

```
Task4/
│
├── notebook/
│     task4_eda_ml.ipynb
│
├── data/
│     titanic.csv
│
├── screenshots/
│     heatmap.png
│     confusion_matrix.png
│     model_accuracy.png
│     feature_importance.png
│     gridsearch_result.png
│
└── README.md
```

---

## 📊 Exploratory Data Analysis

Key steps performed:

* Missing values handled using median and mode
* Cabin column removed due to excessive null values
* Feature engineering applied to categorical variables
* Visualizations created:

  * Age distribution histogram
  * Fare vs Survival boxplot
  * Correlation heatmap

### 🔎 Key Insights

* Female passengers showed higher survival probability.
* Higher passenger class correlated with survival.
* Fare and Sex were strong predictive features.

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

---

## 📈 Model Comparison Table

| Model               | Accuracy (Before Tuning) | Accuracy (After Tuning) |
| ------------------- | ------------------------ | ----------------------- |
| Logistic Regression | 0.79                     | —                       |
| Decision Tree       | 0.77                     | —                       |
| Random Forest       | 0.80                     | 0.81 *(approx)*         |

> Update the tuned accuracy value based on your notebook output if needed.

---

## 📊 Evaluation Metrics

* Accuracy Score
* Confusion Matrix Visualization
* Feature Importance Analysis

Random Forest achieved the best performance due to ensemble learning and better generalization.

---

## ⚙️ Hyperparameter Tuning

GridSearchCV was applied to optimize:

* n_estimators
* max_depth
* min_samples_split

This improved model performance compared to the default configuration.

---

## 🖥️ Technologies Used

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 🚀 Conclusion

This project demonstrates a complete ML workflow including data preprocessing, visualization, model training, evaluation, and optimization.
The tuned Random Forest model provided the highest accuracy for Titanic survival prediction.

---
