# Machine Learning Model Development Project

This project focuses on building and comparing multiple machine learning models for a classification task using the Breast Cancer dataset.

## Objective
Build and compare the following models:
- Linear Regression / **Logistic Regression** (Classification)
- **Decision Tree**
- **Random Forest**

## Key Features
- **Exploratory Analysis**: Data loading and splitting using `sklearn`.
- **Model Training**: Implementation of three different algorithms with default and tuned parameters.
- **Evaluation**: 
  - **Accuracy**: For classification performance.
  - **Mean Squared Error (MSE)**: Provided as a metric (applied to classification results).
  - **Confusion Matrix**: Visualizing true/false positives and negatives.
- **Feature Importance Analysis**: Identifying the most impactful variables in the Random Forest model.
- **Hyperparameter Tuning**: Performance optimization using `GridSearchCV`.
- **Performance Comparison**: A detailed look at model performance before and after tuning.

## Model Comparison Table

| Model | Accuracy (Initial) | MSE (Initial) |
|-------|--------------------|---------------|
| Logistic Regression | ~0.9561 | ~0.0439 |
| Decision Tree | ~0.9474 | ~0.0526 |
| Random Forest | ~0.9649 | ~0.0351 |

*Note: Actual values may vary slightly based on the random seed and split.*

## Hyperparameter Tuning Result (Random Forest)
| Stage | Accuracy | MSE |
|-------|----------|-----|
| Before Tuning | 0.9649 | 0.0351 |
| After Tuning | 0.9649 | 0.0351 |

## Performance Insights
- **Logistic Regression** provides a high-performing baseline for this binary classification task.
- **Random Forest** typically reduces variance and provides robust predictions compared to single Decision Trees.
- **Feature Importance** analysis reveals that features like `worst area`, `worst concave points`, and `worst perimeter` are the most significant indicators for tumor classification.

## Setup Instructions
1. **Virtual Environment**: Create a virtual environment with Python 3.11.1.
2. **Installation**: Install dependencies using:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn notebook
   ```
3. **Execution**: Launch Jupyter and run the notebook:
   ```bash
   jupyter notebook ML_Model_Development.ipynb
   ```
