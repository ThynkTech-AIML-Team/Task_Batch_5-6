Task 1 – Exploratory Data Analysis (EDA)
a)Objective

Perform detailed exploratory analysis to understand patterns affecting passenger survival.

b)Steps Performed

Handled missing values (Age, Embarked)

Dropped highly missing column (Deck)

Created new features (Family Size, Is Alone)

Generated visualizations:

Age histogram

Fare vs Survival boxplot

Correlation heatmap

Survival by gender

c)Insights

Female passengers had significantly higher survival rates.

Higher fare passengers had better survival probability.

Passenger class strongly influenced survival.

Traveling alone slightly reduced survival chances.

Task 2 – Machine Learning Model Development
a)Objective

Build and compare multiple classification models.

b)Models Implemented

Logistic Regression

Decision Tree

Random Forest

c)Evaluation Metrics

Accuracy

Confusion Matrix

Feature Importance

d)Model Comparison Table
Model	Accuracy	Remarks
Logistic Regression	~80%	Stable baseline model
Decision Tree	~78%	Slight overfitting risk
Random Forest	~85%	Best overall performance

Task 3 – Deployment (Streamlit App)
a)Objective

Deploy the trained model as an interactive web application.

b)Features

User inputs:

Passenger Class

Age

Gender

Fare

Family details

Real-time survival prediction

Clean UI using Streamlit
