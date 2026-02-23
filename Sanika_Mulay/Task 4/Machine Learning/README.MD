Task 4 - Task 2: Machine Learning Model Development (House Price Prediction)
Objective
The objective of this task is to develop and compare multiple machine learning models to predict house prices based on various features such as living area, number of bedrooms, bathrooms, and other property characteristics.

This task involves training different regression models, evaluating their performance using Mean Squared Error (MSE), and analyzing feature importance.

Dataset
Dataset used: Cleaned House Price Dataset (from Task 1 - EDA)

The dataset contains numerical features such as:

bedrooms
bathrooms
sqft_living
sqft_lot
floors
waterfront
view
condition
grade
sqft_above
sqft_basement
yr_built
yr_renovated
house_age
total_size
was_renovated
Target variable:

price
Non-numeric columns such as address, id, and date were removed during preprocessing.

Tools and Libraries Used
Python
Jupyter Notebook
Pandas : data manipulation
NumPy : numerical operations
Scikit-learn : machine learning models
Matplotlib : visualization
Seaborn : visualization
Steps Performed
1. Data Loading
Loaded cleaned dataset from Task 1
Verified data types and removed non-numeric columns
2. Feature and Target Selection
Features (X): All independent variables except price

Target (y): price column

3. Train-Test Split
Split dataset into:
80% Training data
20% Testing data
This allows evaluation of model performance on unseen data.

Models Implemented
1. Linear Regression
Basic regression model
Assumes linear relationship between features and target
2. Decision Tree Regressor
Tree-based model
Captures non-linear relationships
3. Random Forest Regressor
Ensemble model using multiple decision trees
Provides better accuracy and reduces overfitting
Model Evaluation Metric
Mean Squared Error (MSE) was used to evaluate model performance. Lower MSE indicates better performance.

Model Performance Comparison
Model	Performance
Linear Regression	Baseline performance
Decision Tree	Improved performance
Random Forest	Best performance (Lowest MSE)
Random Forest performed best due to its ability to capture complex relationships.

Feature Importance Analysis
Feature importance was extracted using Random Forest.

Top important features include:

sqft_living
grade
total_size
bathrooms
house_age
Feature importance helps identify which factors most influence house prices. Feature importance visualization was saved in outputs folder.

Hyperparameter Tuning (Bonus)
GridSearchCV was used to tune Random Forest parameters such as:

n_estimators
max_depth
This improved model performance further.

Conclusion
Multiple machine learning models were successfully trained and evaluated for house price prediction. Random Forest achieved the best performance with the lowest Mean Squared Error. Feature importance analysis identified the most influential variables affecting house price. The trained model is suitable for deployment in the next task.