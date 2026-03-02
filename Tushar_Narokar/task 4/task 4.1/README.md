# Titanic Exploratory Data Analysis (EDA) Project

This project focuses on performing a comprehensive Exploratory Data Analysis (EDA) on the classic Titanic dataset. The goal is to understand the factors that influenced the survival of passengers and to prepare the data for predictive modeling.

## Project Structure
- `Titanic_EDA.ipynb`: Jupyter Notebook containing the full analysis, visualizations, and insights.
- `requirements.txt`: List of dependencies required to run the project.
- `venv/`: Virtual environment with Python 3.11.1.

## Objectives Accomplished
- **Handling Missing Values**: Used Median imputation for Age and dropped non-informative features.
- **Feature Engineering**: Created new features like `FamilySize`, `IsAlone`, and extracted `Title` from passenger names.
- **Visualizations**: 
  - Histograms for Age distribution.
  - Countplots for survival based on Sex and Class.
  - Correlation Heatmap to identify relationships between numerical features.
- **Key Insights**: Identified gender, socio-economic status, and age as primary drivers for survival.

## Model Comparison Table
Although the primary focus is EDA, we've conducted a preliminary baseline modeling to compare performance:

| Model | Accuracy Score | Description |
|-------|----------------|-------------|
| **Logistic Regression** | ~0.81 | Good baseline performance, interpretable results. |
| **Random Forest** | ~0.83 | Better performance due to handling non-linear relationships. |

## Setup Instructions
1. **Virtual Environment**:
   The project uses a virtual environment built with Python 3.11.1.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook**:
   ```bash
   jupyter notebook Titanic_EDA.ipynb
   ```
.\venv\Scripts\jupyter nbconvert --to notebook --execute Titanic_EDA.ipynb --output executed_Titanic_EDA.ipynb