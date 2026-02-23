# 🚢 Titanic Exploratory Data Analysis (EDA)

## 📌 Project Overview
This project performs a comprehensive Exploratory Data Analysis (EDA) on the classic Titanic dataset. The primary objective is to investigate the data, uncover hidden patterns, and determine which factors most significantly influenced the survival rates of passengers aboard the Titanic.

This project is contained entirely within a Jupyter Notebook (`tas1.ipynb`).

## 🎯 Objective
To perform detailed exploratory data analysis on a real-world dataset by cleaning data, engineering new features, and utilizing statistical visualizations to extract actionable insights.

## 📊 Dataset
* **Source:** The Titanic passenger dataset (loaded directly via public URL in the notebook).
* **Description:** Contains demographics and passenger information such as age, sex, passenger class, ticket fare, and survival status.

## 🛠️ Key Tasks Performed

### 1. Data Cleaning & Handling Missing Values
* **Age:** Imputed missing values using the median age to avoid extreme outliers.
* **Embarked:** Filled missing values with the mode (most frequent port of embarkation).
* **Cabin:** Dropped the column due to a high percentage (>75%) of missing data to maintain analysis integrity.

### 2. Feature Engineering
Created several new features to expose deeper relationships in the data:
* **`FamilySize`:** Combined the `SibSp` (siblings/spouses) and `Parch` (parents/children) columns.
* **`Is_Alone`:** A binary indicator representing whether a passenger was traveling solo or with family.
* **`Title`:** Extracted personal titles (e.g., Mr., Mrs., Miss) from the `Name` column using regular expressions and grouped rare titles together.

### 3. Data Visualization
Utilized `matplotlib` and `seaborn` to create clear, informative charts:
* **Histograms:** To view the distribution of passenger ages.
* **Boxplots:** To analyze age distribution relative to survival status.
* **Barplots:** To compare survival probabilities across different sexes and passenger classes.
* **Correlation Heatmap:** To visualize the linear relationships between all numerical variables (e.g., Fare vs. Pclass vs. Survival).

## 💡 Key Insights Discovered
1. **The "Women and Children First" Rule:** Female passengers had a drastically higher survival rate (~74%) compared to male passengers (~19%). Younger passengers also had higher survival rates.
2. **Socio-Economic Status Matters:** First-class passengers had a significantly higher probability of survival compared to third-class passengers.
3. **Fare and Survival:** There is a positive correlation between the ticket fare paid and survival likelihood, which directly ties to the priority given to upper-class passengers.
4. **Family Dynamics:** Passengers traveling completely alone had lower survival rates than those traveling with a small family, though exceptionally large families also faced lower survival probabilities.
