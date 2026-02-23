# Task 4.1 - EDA Project
## Key Insights

## 1. Overview of the House Prediction Dataset

-   The dataset contains **2000 rows** and **10 columns**.
    
-   Features include a mix of **numerical** and **categorical** variables.
    
-   Initial inspection showed no presence of **missing values**, **outliers**, and **data inconsistencies**.
    

----------

## 2. Univariate Analysis Insights

-   The distribution of **Price** is highly skewed, suggesting potential transformation may improve modeling.
    
-   **Area** shows a normal/left/right-skewed distribution.
   
    

----------

## 3. Bivariate & Multivariate Relationships

-   A strong positive/negative correlation exists between **Area** and **Price**.
    
-   The target variable **Price** is strongly influenced by **Area** and **Condition**.

----------

   
## 4. Feature Importance 

-   The most influential features impacting **Price** are:
    
    1.  **Area**
        
    2.  **Location**
        
    3.  **No. of rooms and floors**
    

----------

## 5. Business / Practical Implications

-   Improving **Condition** could significantly impact **Price**.
-   Presence of **Garage** also affects the **Price.**
    

----------

## 6. Conclusion

The exploratory data analysis revealed meaningful relationships, key predictive features, and structural patterns within the dataset. Proper feature engineering like label encoding and improve overall insights derived from the data.

-----
# Task 4.2  -  ML Model Development

## Key Insights
1) Three models trained and evaluated.
2) Tree models provide feature importance plots.
3) GridSearchCV found best hyperparameters shown during visualization.
4) Comparison table created in output, Random Forest shows improvement after tuning.
5) Use tuned Random Forest for best accuracy in this run.