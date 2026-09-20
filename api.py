"""
ChurnGuard AI — FastAPI Prediction Backend with SHAP Explainability
Run with:  python api.py
"""

import warnings
warnings.filterwarnings("ignore")

import uvicorn
import joblib
import shap
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Load model once on startup ──────────────────────────────
model = joblib.load("models/tuned_xgboost_pipeline.pkl")
preprocessor = model.named_steps["preprocessor"]
xgb_clf = model.named_steps["model"]
explainer = shap.TreeExplainer(xgb_clf)

# ── App ─────────────────────────────────────────────────────
app = FastAPI(title="ChurnGuard AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ───────────────────────────────────────────
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def format_feature_name(raw_name: str) -> str:
    cleaned = raw_name.replace("cat__", "").replace("num__", "")
    if "_" in cleaned:
        parts = cleaned.split("_", 1)
        return f"{parts[0]}: {parts[1]}"
    if cleaned == "tenure":
        return "Tenure Length"
    if cleaned == "MonthlyCharges":
        return "Monthly Charges"
    if cleaned == "TotalCharges":
        return "Total Charges"
    if cleaned == "SeniorCitizen":
        return "Senior Citizen Status"
    return cleaned


# ── Endpoints ────────────────────────────────────────────────
@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.post("/predict")
def predict(data: CustomerInput):
    df = pd.DataFrame([data.model_dump()])

    # Prediction & probability
    prediction  = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    if probability >= 0.70:
        risk = "high"
    elif probability >= 0.40:
        risk = "medium"
    else:
        risk = "low"

    # SHAP feature importance calculation
    try:
        transformed = preprocessor.transform(df)
        feature_names = preprocessor.get_feature_names_out()
        shap_res = explainer(transformed)
        shap_vals = shap_res.values[0]
        base_value = float(shap_res.base_values[0]) if hasattr(shap_res, "base_values") else 0.0

        shap_list = []
        for name, val in zip(feature_names, shap_vals):
            v = float(val)
            if abs(v) > 0.005:  # filter negligible features
                shap_list.append({
                    "feature": format_feature_name(name),
                    "raw_feature": name,
                    "value": round(v, 4),
                    "abs_val": round(abs(v), 4),
                    "impact": "churn" if v > 0 else "stay",
                })

        # Sort by absolute magnitude
        shap_list = sorted(shap_list, key=lambda x: x["abs_val"], reverse=True)
        top_shap = shap_list[:10]
    except Exception as e:
        top_shap = []
        base_value = 0.0

    return {
        "prediction":  prediction,
        "probability": round(probability, 4),
        "risk":        risk,
        "base_value":  round(base_value, 4),
        "shap":        top_shap,
    }


# ── Dev runner ───────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
