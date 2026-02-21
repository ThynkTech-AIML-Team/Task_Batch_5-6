Task 4 - Task 3: House Price Prediction Web App Deployment
Objective
The objective of this task is to deploy the trained machine learning model as an interactive web application using Streamlit. The application allows users to input house features and receive a predicted house price based on the trained Random Forest Regression model.

This demonstrates the practical deployment of a machine learning model into a real-world usable interface.

Model Information
The deployed model is a Random Forest Regressor trained in Task 2 using the House Price dataset.

The model was trained using the following features:

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
house_age (engineered feature)
total_size (engineered feature)
was_renovated (engineered feature)
Target variable:

price
The trained model was saved as: model/house_price_model.pkl

Technologies Used
Python
Streamlit
Scikit-learn
NumPy
Pickle
Application Features
The Streamlit web application provides an interactive interface where users can input house characteristics such as:

Number of bedrooms
Number of bathrooms
Living area
Lot area
Number of floors
Waterfront status
View rating
Condition rating
Grade
Sqft above ground
Sqft basement
Year built
Year renovated
The app performs feature engineering and predicts the house price using the trained Random Forest model.

How the Application Works
User enters house details through the web interface
The app performs necessary feature engineering:
house_age
total_size
was_renovated
The trained model predicts the house price
The predicted price is displayed to the user
How to Run the Application
Step 1: Open terminal or Anaconda Prompt
Navigate to the app folder: cd "Task 4/Task 3 - Deployment/app"

Step 2: Run Streamlit
streamlit run app.py

Step 3: Open browser
The app will automatically open at: http://localhost:8501

Output
The application successfully predicts house prices based on user input and provides an interactive user interface.

Screenshots of the running application are included as part of submission.

Conclusion
This task demonstrates successful deployment of a machine learning model into a real-time prediction system using Streamlit. The model can now be used by end users to estimate house prices based on property features.

This completes the full machine learning pipeline:

Data Analysis
Model Development
Model Deployment