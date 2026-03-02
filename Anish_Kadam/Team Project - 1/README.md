# 🔧 Predictive Maintenance System (Time-Series)

## 🚀 Project Overview
This project implements an industry-grade predictive maintenance system that predicts equipment failure using time-series sensor data.

The system leverages deep learning (LSTM) to capture temporal degradation patterns and optimize failure probability thresholds for early breakdown detection.

---

## 🎯 Objective
Predict equipment failure using sequential sensor data and provide reliable early warning alerts.

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn

---

## 📊 Methodology

### 1️⃣ Data Processing
- Time-series safe split (no leakage)
- Rolling statistics (mean, std)
- Lag features
- Rate-of-change features

### 2️⃣ Baseline Model
- Random Forest classifier
- Feature importance analysis

### 3️⃣ Deep Time-Series Modeling
- 30-step sequence window
- Stacked LSTM architecture
- Batch normalization + Dropout
- EarlyStopping regularization

### 4️⃣ Failure Probability Optimization
- Precision-Recall curve
- F1-score based threshold tuning
- Improved early failure detection

---

## 📈 Performance Metrics
- Accuracy
- F1 Score
- Mean Squared Error (MSE)
- ROC-AUC

LSTM achieved superior temporal modeling performance compared to baseline ML models.

---

## 📊 Visualization
- ROC Curve comparison
- Feature importance analysis
- Time-series breakdown visualization 

---

## 💾 Deployment Ready
- Trained models saved (.pkl, .h5)
- Scaler saved for production usage 

---

## 👤 Team 2 - Sarthak Nagave, Athrva Admile, Lalit More, Ankita Kakade, Sinchan Rao, Venkatesh Gudade, Anish Kadam

**Anish Kadam**  
Packaging trained model, saving, kept ready for deployment.
