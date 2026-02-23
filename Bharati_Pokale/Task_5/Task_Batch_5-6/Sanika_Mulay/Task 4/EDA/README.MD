Task 4 - Task 1: Exploratory Data Analysis (EDA) on House Price Dataset
Objective
The objective of this task is to perform Exploratory Data Analysis (EDA) on a real-world house price dataset to understand the structure of the data, identify important features, handle missing values, perform feature engineering, and extract meaningful insights through visualization and statistical analysis.

Dataset
Dataset used: House Price Prediction Dataset

The dataset contains various features related to residential homes such as:

price (Target variable)
bedrooms
bathrooms
sqft_living
sqft_lot
floors
waterfront
view
condition
grade
yr_built
yr_renovated
and other related features
Tools and Libraries Used
Python
Jupyter Notebook
Pandas – data manipulation
NumPy – numerical operations
Matplotlib – data visualization
Seaborn – advanced visualization
Steps Performed
1. Data Loading and Inspection
Loaded dataset using Pandas
Examined dataset structure using:
head()
shape
info()
describe()
2. Missing Value Handling
Identified missing values using isnull()
Filled missing numeric values using median
Filled missing categorical values using mode
3. Feature Engineering
Created new meaningful features to improve analysis:

house_age = current_year - yr_built
total_size = sqft_living + sqft_lot
was_renovated = indicator for renovation status
These features help better understand house value relationships.

4. Data Visualization
The following visualizations were created:

Price Distribution Histogram
Price Boxplot (to detect outliers)
Scatter Plot (sqft_living vs price)
Correlation Heatmap (feature relationships)
These visualizations helped identify important trends and patterns.

5. Correlation Analysis
Analyzed correlation between features and house price to identify the most influential variables affecting house price.

Key strongly correlated features include:

sqft_living
bathrooms
grade
total_size
6. Key Insights
Important observations from the dataset:

Houses with larger living area have significantly higher prices
Total house size strongly affects price
Newer houses tend to have higher prices
Renovated houses generally have higher prices
Some extreme outliers exist in price distribution
Output Files
The following outputs were generated and saved:

Cleaned dataset: outputs/house_price_cleaned.csv
Visualizations:
price_histogram.png
price_boxplot.png
scatterplot.png
correlation_heatmap.png