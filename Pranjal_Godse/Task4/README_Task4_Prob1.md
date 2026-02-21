# 🚢 Task 4 -- Data Science Project

Dataset: Titanic Dataset

------------------------------------------------------------------------

## 📌 Project Overview

This project includes:

1.  Exploratory Data Analysis (EDA)\
2.  Machine Learning Model Development\
3.  Model Comparison and Evaluation

The objective is to analyze the Titanic dataset and build multiple
machine learning models to predict passenger survival.

------------------------------------------------------------------------

## 🟢 1. Exploratory Data Analysis (EDA)

### ✔ Steps Performed:

-   Handled missing values\
-   Performed feature engineering\
-   Created visualizations:
    -   Histogram\
    -   Boxplot\
    -   Correlation heatmap\
-   Extracted key insights

### ✔ Key Insights:

-   Female passengers had higher survival rate.\
-   First-class passengers survived more.\
-   Higher fare is positively correlated with survival.\
-   Family size influences survival probability.\
-   Gender and Pclass are strong predictors.

------------------------------------------------------------------------

## 🟢 2. Machine Learning Models Used

-   Logistic Regression\
-   Decision Tree\
-   Random Forest

------------------------------------------------------------------------

## 🟢 3. Evaluation Metrics Used

-   Accuracy\
-   Confusion Matrix\
-   Feature Importance

------------------------------------------------------------------------

## 🟢 4. Model Comparison Table

  ------------------------------------------------------------------------------
  Model              Accuracy (Before       Accuracy (After        Remarks
                     Tuning)                Tuning)                
  ------------------ ---------------------- ---------------------- -------------
  Logistic           80%                    82%                    Good baseline
  Regression                                                       model

  Decision Tree      78%                    83%                    Overfitting
                                                                   reduced after
                                                                   tuning

  Random Forest      84%                    87%                    Best
                                                                   performing
                                                                   model
  ------------------------------------------------------------------------------

------------------------------------------------------------------------

## 🟢 5. Best Model

Random Forest performed the best with highest accuracy and better
generalization after hyperparameter tuning.

------------------------------------------------------------------------

## 🟢 6. Project Structure

Task_4/ │ ├── Titanic_EDA.ipynb\
├── Model_Development.ipynb\
├── README.md\
└── screenshots/

------------------------------------------------------------------------

## 🟢 7. Tools & Technologies Used

-   Python\
-   Pandas\
-   NumPy\
-   Matplotlib\
-   Seaborn\
-   Scikit-learn\
-   Jupyter Notebook

------------------------------------------------------------------------

## 🟢 8. Conclusion

EDA and model comparison show that ensemble methods like Random Forest
provide better prediction performance for Titanic survival
classification.
