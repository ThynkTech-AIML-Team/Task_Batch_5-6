import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Create a dummy dataset for House Prices
data = {
    'Size_sqft': [1500, 2000, 2500, 1200, 3000, 1800, 2200, 2800],
    'Bedrooms': [3, 4, 4, 2, 5, 3, 4, 5],
    'Age_years': [10, 5, 2, 15, 1, 8, 4, 3],
    'Price': [300000, 450000, 500000, 200000, 650000, 350000, 480000, 550000]
}
df = pd.DataFrame(data)

# 2. Split features and target
X = df.drop('Price', axis=1)
y = df['Price']

# 3. Train a Regression Model
print("Training the model...")
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# 4. Save the model to a file
joblib.dump(model, 'house_price_model.pkl')
print("Model saved successfully as 'house_price_model.pkl'!")