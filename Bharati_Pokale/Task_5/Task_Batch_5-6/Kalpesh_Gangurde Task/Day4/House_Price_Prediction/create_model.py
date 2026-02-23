import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Create sample training data
# Features: [Area, Bedrooms, Bathrooms, Location]
X_train = np.array([
    [2000, 3, 2, 0],      # Downtown
    [1500, 2, 1.5, 1],    # Suburbs
    [3500, 4, 3, 0],      # Downtown
    [1200, 2, 1, 2],      # Rural (50 Lakh)
    [4000, 5, 4, 3],      # Waterfront
    [2500, 3, 2.5, 1],    # Suburbs
    [1800, 3, 2, 2],      # Rural
    [3200, 4, 3, 0],      # Downtown
    [2800, 3, 2.5, 3],    # Waterfront (2.5 Cr)
    [1600, 2, 1.5, 1],    # Suburbs
])

# Sample prices (in Indian Rupees)
y_train = np.array([
    13500000,  # 2000 sqft, 3br, 2ba, Downtown - 1.35 Cr
    8400000,   # 1500 sqft, 2br, 1.5ba, Suburbs - 84 Lakh
    19500000,  # 3500 sqft, 4br, 3ba, Downtown - 1.95 Cr
    5000000,   # 1200 sqft, 2br, 1ba, Rural - 50 Lakh
    27000000,  # 4000 sqft, 5br, 4ba, Waterfront - 2.7 Cr
    11400000,  # 2500 sqft, 3br, 2.5ba, Suburbs - 1.14 Cr
    9600000,   # 1800 sqft, 3br, 2ba, Rural - 96 Lakh
    17400000,  # 3200 sqft, 4br, 3ba, Downtown - 1.74 Cr
    25000000,  # 2800 sqft, 3br, 2.5ba, Waterfront - 2.5 Cr
    8700000,   # 1600 sqft, 2br, 1.5ba, Suburbs - 87 Lakh
])

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
print("Model created and saved as 'model.pkl'")
print(f"Model R² score on training data: {model.score(X_train, y_train):.4f}")
