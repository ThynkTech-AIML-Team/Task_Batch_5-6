# 🚢 Task 4 -- Problem 2

## Machine Learning Model Development

Dataset: Titanic Dataset

------------------------------------------------------------------------

## 📌 Objective

Build and compare multiple machine learning models to predict passenger
survival.

Models Implemented: - Logistic Regression - Decision Tree - Random
Forest

------------------------------------------------------------------------

## 🟢 Steps Performed

1.  Data Preprocessing
    -   Handled missing values
    -   Feature engineering (FamilySize, IsAlone)
    -   Encoding categorical variables
    -   Train-test split (80-20)
2.  Model Training
    -   Logistic Regression (Baseline Model)
    -   Decision Tree
    -   Random Forest
3.  Model Evaluation
    -   Accuracy Score
    -   Confusion Matrix
    -   Feature Importance (Random Forest)
4.  Hyperparameter Tuning (Bonus)
    -   Applied GridSearchCV
    -   Tuned Decision Tree
    -   Tuned Random Forest
    -   Compared performance before and after tuning

------------------------------------------------------------------------

## 🟢 Model Comparison Table

  ---------------------------------------------------------------------------
  Model                     Before Tuning     After Tuning      Remarks
  ------------------------- ----------------- ----------------- -------------
  Logistic Regression       \~80%             \~82%             Strong
                                                                baseline
                                                                model

  Decision Tree             \~78%             \~83%             Overfitting
                                                                reduced after
                                                                tuning

  Random Forest             \~84%             \~87%             Best
                                                                performing
                                                                model
  ---------------------------------------------------------------------------

------------------------------------------------------------------------

## 🟢 Evaluation Metrics Used

-   Accuracy
-   Confusion Matrix
-   Feature Importance

(Note: Mean Squared Error is applicable for regression problems. Since
Titanic is a classification problem, Accuracy is used.)

------------------------------------------------------------------------

## 🟢 Best Model

Random Forest achieved the highest accuracy and better generalization
after hyperparameter tuning.

------------------------------------------------------------------------

## 🟢 Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Jupyter Notebook

------------------------------------------------------------------------

## 🟢 Conclusion

Ensemble learning methods like Random Forest provide better predictive
performance compared to individual models like Logistic Regression and
Decision Tree. Hyperparameter tuning significantly improves model
performance.
