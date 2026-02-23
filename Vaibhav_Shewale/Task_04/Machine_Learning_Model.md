# Machine Learning Model Comparison

**Models compared**  
- Linear Regression  
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Task types covered**  
- Regression → Linear Regression, Decision Tree Regressor, Random Forest Regressor  
- Classification → Logistic Regression, Decision Tree Classifier, Random Forest Classifier  

**Evaluation Metrics**  
- Accuracy (classification)  
- Mean Squared Error – MSE (regression)  
- Confusion Matrix (classification)  
- Feature Importance  

Last updated: February 2025

## 1. Quick Overview Table

| Criterion                  | Linear Regression          | Logistic Regression        | Decision Tree              | Random Forest                  |
|----------------------------|----------------------------|----------------------------|----------------------------|--------------------------------|
| **Primary Task**           | Regression                | Classification            | Both                      | Both                          |
| **Model Type**             | Parametric, Linear        | Parametric, Linear        | Non-parametric, Tree      | Ensemble (Bagging)            |
| **Interpretability**       | ★★★★★ (very high)        | ★★★★☆ (high)             | ★★★★☆ (good – visual)    | ★★☆☆☆ (moderate)             |
| **Handles Non-linearity**  | Poor                      | Poor                      | Good                      | Excellent                     |
| **Overfitting Risk**       | Low                       | Low                       | High                      | Low                           |
| **Training Speed**         | Very fast                 | Very fast                 | Fast                      | Moderate–Slow                 |
| **Prediction Speed**       | Very fast                 | Very fast                 | Fast                      | Moderate                      |
| **Feature Scaling Needed** | Usually yes               | Usually yes               | No                        | No                            |
| **Robust to Outliers**     | Poor                      | Poor                      | Moderate                  | Good                          |
| **Typical Performance**    | Baseline                  | Good on linear problems   | Good baseline             | Often best among these four   |

## 2. Evaluation Metrics – How They Behave

### Classification Tasks

| Metric              | Logistic Regression                  | Decision Tree                        | Random Forest                          |
|---------------------|--------------------------------------|--------------------------------------|----------------------------------------|
| **Accuracy**        | Good on linearly separable data      | Can be high (but often overfits)     | Usually highest & most stable          |
| **Confusion Matrix**| Balanced if classes are separable    | May have many false predictions in leaves with few samples | Fewer off-diagonal errors, more balanced |
| **Best When**       | Data is roughly linear in log-odds   | Need quick interpretable model       | Highest predictive power is priority   |

### Regression Tasks

| Metric              | Linear Regression                    | Decision Tree Regressor              | Random Forest Regressor                |
|---------------------|--------------------------------------|--------------------------------------|----------------------------------------|
| **MSE / RMSE**      | Optimal *if* relationship is linear  | Can be much lower than linear model on non-linear data | Usually lowest MSE among the four      |
| **Best When**       | Data has clear linear trend          | Non-linear patterns, interactions    | Complex patterns + want robustness     |

## 3. Feature Importance Comparison

| Model                | How Feature Importance is Calculated                          | Signed? | Reliability | Typical Ranking Quality |
|----------------------|----------------------------------------------------------------|--------|-------------|--------------------------|
| **Linear / Logistic**| Coefficient magnitude (absolute value)                        | Yes    | High        | Good if features are scaled |
| **Decision Tree**    | Total reduction in impurity (Gini / Entropy / MSE) across splits | No     | Moderate    | Can overestimate correlated features |
| **Random Forest**    | Mean decrease in impurity across all trees                     | No     | High        | Most reliable among tree-based models |

**Quick rule of thumb**:
- Want **signed** importance (positive/negative effect) → use **Linear / Logistic Regression**
- Want most **trustworthy** importance ranking → use **Random Forest**

## 4. When to Choose Which Model (Decision Guide)

```text
Is interpretability the #1 priority?
├── Yes ──> Logistic / Linear Regression  (or single Decision Tree with max_depth ≤ 4–5)
└── No
    ├── Small/medium dataset, want fast training & good performance?
    │   └── Decision Tree (with proper pruning)
    └── Best possible performance matters most?
        └── Random Forest (almost always the strongest among these four)

Data is clearly linear?                    → Start with Linear / Logistic Regression
Data has clear non-linear patterns?        → Skip linear models, go to trees
Need model to run on very low resources?   → Linear / Logistic or pruned Decision Tree
Want best out-of-the-box performance?      → Random Forest