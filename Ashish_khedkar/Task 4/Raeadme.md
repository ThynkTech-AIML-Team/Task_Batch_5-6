
## Part 1 & Part 2 – EDA and Model Building

* Dataset used: Titanic (seaborn built-in)
* Handled missing values in age and embarked columns
* Created a new feature: `family_size`
* Performed EDA using:

  * Histogram
  * Boxplot
  * Correlation heatmap
* Models used:

  * Logistic Regression
  * Decision Tree
  * Random Forest

### Model Accuracy (approx)

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~78%     |
| Decision Tree       | ~75%     |
| Random Forest       | ~82%     |

Notebook file:
`Part_1 and 2.ipynb`

---

## Part 3 – Titanic Survival Predictor (Streamlit)

A simple web app that predicts Titanic passenger survival based on user inputs.

* Model: Random Forest
* Inputs: class, age, sex, fare, embark location, family size

Run the app:

```bash
python -m streamlit run app.py
```
