# 🛡️ ChurnGuard AI · Customer Churn Prediction & Explainability System

[![Streamlit](https://img.shields.io/badge/Streamlit-1.47+-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=for-the-badge&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-EB5424.svg?style=for-the-badge&logo=XGBoost&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.45+-blueviolet.svg?style=for-the-badge)](https://shap.readthedocs.io)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)](https://python.org)

> **ChurnGuard AI** is an enterprise-grade customer churn prediction and risk analytics platform powered by a tuned **XGBoost** model with real-time **TreeSHAP** explainability and the **Sentinel Dark v2.0** command-center dashboard.

---

## 📸 Overview & Features

![Sentinel Dark UI](https://raw.githubusercontent.com/YASWINKUMAR02/Customer_Churn_Prediction/main/outputs/preview.png)

* **🤖 Machine Learning Pipeline**: Trained on customer demographic, service, and billing patterns using a robust `imblearn` pipeline combining `StandardScaler`, `OneHotEncoder`, `SMOTE` class balancing, and a tuned `XGBClassifier`.
* **🧠 Real-Time SHAP Explainability**: Implements `shap.TreeExplainer` to compute local feature attributions on every prediction, surfacing exact factors accelerating churn (🔴 **Churn Drivers**) and factors preserving customer loyalty (🟢 **Retention Anchors**).
* **🎨 Sentinel Dark v2.0 Interface**: Cyber-themed command-center with segmented selectors, animated toggle switches, smooth multi-tier risk meter gauge, and automated attribution summaries.
* **⚡ Dual Serving Architecture**:
  * **Streamlit App (`app.py`)**: 100% self-contained, cloud-ready for instant deployment on Streamlit Community Cloud or Hugging Face Spaces.
  * **FastAPI Backend (`api.py`) + Client (`index.html`)**: Production REST API endpoint (`POST /predict`) with standalone single-page web client.

---

## 📁 Repository Structure

```
Customer_Churn_Prediction/
├── api.py                     # FastAPI REST API backend with SHAP endpoint
├── app.py                     # Self-contained Streamlit Cloud application
├── index.html                 # Sentinel Dark standalone web interface
├── requirements.txt           # Python dependencies for local & cloud deployment
├── README.md                  # Project documentation
├── data/                      # Dataset directory (Telco Customer Churn)
├── models/
│   └── tuned_xgboost_pipeline.pkl  # Serialized end-to-end XGBoost model
├── notebooks/                 # Exploratory data analysis & model training notebooks
└── outputs/                   # Visualizations, charts, and metrics
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YASWINKUMAR02/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### 2. Create a virtual environment & install dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Streamlit Dashboard (Recommended)
Runs the self-contained dashboard with native Python model inference and SHAP explainability:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

### Option 2: FastAPI Backend + Web Client
Runs the REST API and serves the single-page application on port 8000:
```bash
python api.py
```
* **Web UI**: Open **[http://localhost:8000](http://localhost:8000)**
* **Interactive API Docs (Swagger UI)**: Open **[http://localhost:8000/docs](http://localhost:8000/docs)**

#### Example REST API Call (`POST /predict`):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Male",
       "SeniorCitizen": 0,
       "Partner": "No",
       "Dependents": "No",
       "tenure": 12,
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "DSL",
       "OnlineSecurity": "No",
       "OnlineBackup": "No",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "No",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "No",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 70.0,
       "TotalCharges": 1000.0
     }'
```

---

## 🌐 Deploying to Streamlit Community Cloud

1. Fork or push this repository to your GitHub account.
2. Visit **[share.streamlit.io](https://share.streamlit.io/)** and sign in.
3. Click **"New app"**, select your repository, set the **Main file path** to `app.py`, and click **Deploy**.

---

## 📊 Features & Model Details

The model evaluates **19 customer attributes**:

| Category | Features |
| :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account & Tenure** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| **Connectivity & Services** | `PhoneService`, `MultipleLines`, `InternetService` |
| **Add-on Subscriptions** | `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## 🛡️ License

Distributed under the **MIT License**. See `LICENSE` for more information.
