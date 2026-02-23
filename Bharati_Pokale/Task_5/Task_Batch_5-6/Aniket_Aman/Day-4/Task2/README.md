# 🧠 Machine Learning Model Development: Breast Cancer Classification

## 📌 Project Overview
This project focuses on building, evaluating, and comparing multiple machine learning models to solve a binary classification problem. The goal is to accurately classify breast cancer tumors as either malignant or benign based on the physical characteristics of cell nuclei. 

This project demonstrates the end-to-end machine learning workflow, from data preprocessing to model evaluation and hyperparameter tuning, contained entirely within a Jupyter Notebook (`task2.ipynb`).

## 🎯 Objective
To develop and compare the performance of different machine learning algorithms, extract feature importance, and optimize model performance using cross-validation techniques.

## 📊 Dataset
* **Source:** The built-in Breast Cancer dataset from `scikit-learn`.
* **Description:** Contains 569 instances with 30 numeric, predictive features (e.g., radius, texture, perimeter, area, smoothness) computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## 🛠️ Models Implemented
1. **Logistic Regression** (Baseline Model)
2. **Decision Tree Classifier**
3. **Random Forest Classifier** (Ensemble Model)

## 📈 Evaluation Metrics Used
* **Accuracy Score:** To measure the overall correctness of the models.
* **Confusion Matrix:** To visualize the true positives, true negatives, false positives, and false negatives for each model.
* **Feature Importance Analysis:** To interpret the Random Forest model and identify which cell features are most indicative of malignancy.

## ⚙️ Key Workflow Steps
1. **Data Preprocessing:** * Split the dataset into training (80%) and testing (20%) sets using stratified sampling.
   * Applied `StandardScaler` to normalize the features, ensuring algorithms like Logistic Regression perform optimally.
2. **Baseline Training & Evaluation:** Trained all three models on the default parameters and evaluated their out-of-the-box accuracy and confusion matrices.
3. **Feature Importance:** Extracted and plotted the top 10 most influential features using the trained Random Forest model.
4. **Hyperparameter Tuning (Bonus):** Applied `GridSearchCV` with 5-fold cross-validation to the Decision Tree model to search for the optimal combination of `max_depth`, `min_samples_split`, and `criterion`.

## 💡 Key Results & Insights
* **Baseline Performance:** Logistic Regression performed exceptionally well as a baseline (~97% accuracy), proving that simple linear models are highly effective when data is cleanly scaled and somewhat linearly separable.
* **Feature Importance:** The Random Forest model revealed that the "worst" dimensions of the cell nuclei (specifically *worst radius*, *worst perimeter*, and *worst area*) are the most critical indicators for predicting malignancy.
* **Model Optimization:** The default Decision Tree showed signs of overfitting. By using `GridSearchCV` to constrain the tree's depth and splitting criteria, the model's accuracy on the unseen test set improved noticeably.
