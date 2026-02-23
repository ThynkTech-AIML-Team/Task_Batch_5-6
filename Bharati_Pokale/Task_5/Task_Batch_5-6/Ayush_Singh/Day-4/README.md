# Task 4 – Titanic Survival Prediction Project

**AIML Internship – ThynkTech**

## 📌 Project Overview

This project implements a complete Machine Learning pipeline on the Titanic dataset, including:

* Exploratory Data Analysis (EDA)
* Machine Learning Model Development and Comparison
* Hyperparameter Tuning
* Model Saving
* Deployment using Streamlit Web App

The goal is to predict whether a passenger survived or not based on their personal and travel details.

---

## 📂 Project Structure

```
Day-4/
│
├── notebook/
│   └── task4_titanic_complete.ipynb     # Complete notebook (EDA + ML + Deployment prep)
│
├── data/
│   └── titanic.csv                     # Original dataset
│
├── outputs/
│   ├── processed_data/
│   │   ├── titanic_cleaned.csv        # Cleaned dataset
│   │   └── best_model.pkl             # Trained ML model
│   │
│   ├── plots/
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   │
│   └── screenshots/
│       ├── eda_dataset_preview.png
│       ├── eda_correlation_heatmap.png
│       ├── ml_model_accuracy_comparison.png
│       ├── ml_confusion_matrix.png
│       ├── ml_feature_importance.png
│       ├── ml_hyperparameter_tuning.png
│       ├── app_ui.png
│       └── app_prediction_result.png
│
├── app/
│   └── app.py                         # Streamlit deployment app
│
└── README.md
```

---

## 📊 Task-1: Exploratory Data Analysis (EDA)

### Steps Performed:

* Loaded and explored Titanic dataset
* Handled missing values using median and mode
* Performed feature engineering (FamilySize)
* Converted categorical variables into numeric format
* Created visualizations:

  * Histogram
  * Boxplot
  * Correlation Heatmap

### Key Insights:

* Female passengers had higher survival rate
* Passenger class strongly influenced survival
* Fare and survival showed positive correlation
* Family size impacted survival probability

---

## 🤖 Task-2: Machine Learning Model Development

### Models Implemented:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### Evaluation Metrics Used:

* Accuracy Score
* Confusion Matrix
* Feature Importance Analysis

### Model Comparison:

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~82%     |
| Decision Tree       | ~78%     |
| Random Forest       | ~87%     |

Random Forest performed the best.

---

## ⚙️ Hyperparameter Tuning

GridSearchCV was used to optimize Random Forest parameters:

* n_estimators
* max_depth

This improved model performance and generalization.

Best model saved as:

```
outputs/processed_data/best_model.pkl
```

---

## 🌐 Task-3: Deployment using Streamlit

A web application was built using Streamlit to predict survival interactively.

### Features:

* User-friendly interface
* Real-time prediction
* Input fields:

  * Passenger Class
  * Sex
  * Age
  * Fare
  * Family Size
  * Embarked Location

### Run the app:

```
cd Day-4/app
python -m streamlit run app.py
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Pickle

---

## 📈 Output Results

The project successfully:

* Cleaned and analyzed dataset
* Built and compared ML models
* Selected best performing model
* Deployed interactive prediction system

---

## 🎯 Learning Outcomes

* Data preprocessing and cleaning
* Feature engineering techniques
* Machine learning model training
* Model evaluation and comparison
* Hyperparameter tuning
* Model deployment using Streamlit
* End-to-end ML project implementation

---

## 👨‍💻 Author

Ayush Singh
AIML Intern – ThynkTech

---

## ✅ Internship Task Completion Status

| Task                          | Status    |
| ----------------------------- | --------- |
| EDA                           | Completed |
| ML Model Development          | Completed |
| Model Comparison              | Completed |
| Hyperparameter Tuning         | Completed |
| Deployment                    | Completed |
| Screenshots and Documentation | Completed |

---

## 🚀 Conclusion

This project demonstrates a complete end-to-end Machine Learning workflow, from data analysis to deployment, and provides a fully functional survival prediction web application.
