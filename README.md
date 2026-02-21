# 📉 Telco Customer Churn Prediction Project

A complete end-to-end Machine Learning & MLOps project to predict customer churn using CatBoost, with full lifecycle management including MLflow tracking, model registry, FastAPI deployment, and Streamlit dashboard visualization.

This project is designed from a Production-Ready ML Engineer perspective, focusing on scalability, reproducibility, monitoring, and real-world deployment standards.

---

## 🚀 Project Overview

The goal of this project is to classify customers into:

- **0 → Stayed**
- **1 → Churned**

Using behavioral, demographic, service usage, and billing features from the Telco dataset.

The system includes:

✔ Data Pipeline & Feature Engineering  
✔ Basic & Advanced EDA  
✔ CatBoost Model with Cross Validation  
✔ Threshold Optimization  
✔ MLflow Tracking & Model Registry  
✔ FastAPI Production API  
✔ Database Logging  
✔ Rate Limiting & JWT Support  
✔ Streamlit Dashboard  
✔ Observability (Logging & Monitoring)

---

## 🗂 Project Architecture

```
Data → Cleaning → EDA → Feature Engineering → Model Training → 
Threshold Optimization → MLflow Tracking → Model Registry → 
FastAPI API → Database Logging → Streamlit Dashboard
```

---

## 📊 Dataset

Telco Customer Churn Dataset  
Target Variable: **Churn (0 = Stay, 1 = Leave)**

Key Features:

- Contract Type
- Tenure
- Internet Service
- Monthly Charges
- Total Charges
- Payment Method
- TechSupport + OnlineSecurity Combination
- Number of Services

---

## 🧹 Data Pipeline

File: `data_pipeline.py`

### Steps:

- Data Loading
- Duplicate & Missing Value Check
- Type Conversion (TotalCharges → numeric)
- Binary Encoding (Yes/No → 0/1)
- Feature Engineering:
  - NumServices
  - TechSupport_OnlineSecurity
- Skewness Analysis
- √ Transformation on TotalCharges
- IQR Outlier Detection
- Distribution Visualization

✔ All numerical features treated  
✔ No outliers after transformation  
✔ Skewness normalized (~0.3)

---

## 📈 Exploratory Data Analysis

### Basic EDA (`basic_eda.py`)

- Gender distribution
- Senior Citizen percentage
- Contract distribution
- Payment method distribution
- Tenure analysis
- Monthly & Total charges distribution
- Churn imbalance (26.5%)

### Advanced EDA (`advanced_eda.py`)

Insights discovered:

- Month-to-month contracts churn the most
- Fiber Optic users churn more than DSL
- Customers without Tech Support churn significantly more
- Short tenure + high charges = highest churn risk
- Senior citizens slightly higher churn rate
- Electronic check users churn more

---

## 🤖 Machine Learning Model

File: `model.py`

### Model: CatBoostClassifier

### Key Features:

- Stratified K-Fold (5 folds)
- Class imbalance handling (scale_pos_weight)
- Early Stopping
- Threshold Optimization (F1 maximization)
- Feature Importance
- Calibration Curve

---

## 📊 Final Model Performance

| Metric | Score |
|--------|--------|
| Accuracy | 0.7658 |
| AUC | 0.8456 |
| Best F1 Score | 0.6341 |
| CV AUC Mean | 0.8484 |
| CV AUC Std | 0.0105 |
| Precision (Churn) | 0.5417 |
| Recall (Churn) | 0.7647 |

✔ Model passes Quality Gate  
(AUC ≥ 0.80 & Recall ≥ 0.70)

---

## 🔁 MLflow Lifecycle Management

File: `mlflow_lifecycle.py`

Features:

- Experiment Tracking
- Parameter Logging
- Metric Logging
- Artifact Logging (plots, JSON reports)
- Model Signature Inference
- PyFunc Wrapper
- Model Registry
- Automatic Stage Promotion
  - Staging → Production (Quality Gate Based)

Registered Model:
```
Churn_Predictor_PyFunc
```

---

## 🌐 FastAPI Production API

File: `api.py`

Features:

- MLflow PyFunc model loading
- Feature Adapter Layer
- Rate Limiting (10 requests/min)
- CORS Protection
- JWT-aware smart identification
- SQLAlchemy Database Logging
- Background Tasks
- Execution Time Header
- CRUD for Predictions

Endpoints:

- POST `/predict`
- GET `/predictions`
- GET `/predictions/{id}`
- DELETE `/predictions/{id}`
- GET `/` (Health Check)

---

## 🖥 Streamlit Dashboard

File: `app.py`

Features:

- Risk Gauge Visualization
- Probability Indicator
- Decision Display (Stay / Churn)
- Global Retention Dashboard
- Pie Chart (Churn Distribution)
- Scatter Plot (Tenure vs Charges)
- Live Data from API

---

## 📦 MLflow Project Configuration

Project Name:
```
Telco_Churn_Prediction
```

Supports parameter tuning:

```
--iterations
--learning_rate
--depth
```

---

## 📡 Observability (Logging & Monitoring)

✔ Structured Logging  
✔ Background Database Logging  
✔ Prediction History Storage  
✔ Model Performance Tracking  
✔ Latency Measurement  
✔ Quality Gate Monitoring  

Logs stored in:
```
app_activity.log
```

Database Tables:

- customer_predictions
- prediction_logs

---

## ⚙️ Environment

Python 3.9

Libraries:

- CatBoost
- MLflow
- FastAPI
- Streamlit
- SQLAlchemy
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 🏁 How to Run

### 1️⃣ Train + Track Model

```
mlflow run . -P iterations=1500 -P learning_rate=0.03 -P depth=6
```

### 2️⃣ Run FastAPI

```
uvicorn api:app --reload
```

### 3️⃣ Run Streamlit Dashboard

```
streamlit run app.py
```

---

## 🎯 Production-Ready Highlights

✔ End-to-End ML Pipeline  
✔ Full MLOps Lifecycle  
✔ Model Registry & Promotion  
✔ API Security & Rate Limiting  
✔ Database Logging  
✔ Real-Time Dashboard  
✔ Threshold Optimization  
✔ Calibration Monitoring  

---

## 👩‍💻 Author

Machine Learning Engineer  
Customer Retention & Predictive Analytics Focus  

---

## ⭐ Project Goal

To demonstrate a complete real-world ML system —  
from raw data to production API —  
with monitoring, governance, and deployment best practices.
