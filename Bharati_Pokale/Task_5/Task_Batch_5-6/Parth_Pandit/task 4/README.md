Titanic Survival Predictor

Project Overview:
This project is an end-to-end Machine Learning application that predicts whether a passenger would survive the Titanic disaster based on demographic and travel-related features.  
The project covers **Exploratory Data Analysis (EDA)**, **Machine Learning model development**, and **deployment using Streamlit**.

Objectives:
- Perform Exploratory Data Analysis on the Titanic dataset
- Build and compare multiple Machine Learning models
- Apply feature engineering to improve predictions
- Deploy the best-performing model using an interactive web interface

Project Structure:
├── EDA_and_Model_Training.ipynb # Task 1 & Task 2 (EDA + ML models)
├── app.py # Streamlit deployment app
├── titanic_survival_model.pkl # Trained Random Forest model
├── README.md # Project documentation
└── Outputs/ # App output screenshots


---

Dataset:
- **Dataset Used:** Titanic Dataset  
- **Source:** Seaborn built-in dataset  
- **Size:** ~900 rows, lightweight and memory efficient  

## Task 1: Exploratory Data Analysis (EDA)
The following steps were performed:
- Dataset inspection and statistical summary
- Handling missing values
- Encoding categorical variables
- Feature engineering
- Data visualization:
  - Histograms
  - Boxplots
  - Correlation heatmap
- Extraction of key insights

### Key Insights:
- Females had significantly higher survival rates
- Passenger class strongly influenced survival
- Children had better survival chances
- Fare and family size played an important role

## Feature Engineering
The following features were engineered:
- **Family Size** = siblings/spouses + parents/children + 1
- **Is Alone** = whether the passenger was traveling alone
- **Fare per Person** = fare divided by family size

These features help capture social and economic factors affecting survival.

## Task 2: Machine Learning Models
Three classification models were trained and evaluated:

| Model | Accuracy (Approx) |
|------|------------------|
| Logistic Regression | ~78–80% |
| Decision Tree | ~75–78% |
| Random Forest | ~80–82% |

### Evaluation Metrics:
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score
- Feature Importance

### Hyperparameter Tuning:
- GridSearchCV was applied to the Random Forest model
- Tuning improved generalization and reduced overfitting

**Final Selected Model:** Random Forest Classifier

## Task 3: Deployment – Titanic Survival Predictor
An interactive web application was built using **Streamlit**.

### Features:
- User-friendly input form
- Real-time survival prediction
- Clear result display (Survived / Not Survived)

### Input Parameters:
- Passenger Class
- Gender
- Age
- Fare Paid
- Family Size

- Port of Embarkation
