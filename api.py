"""
ChurnGuard AI — FastAPI Prediction Backend
Run with:  python api.py
"""

import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Load model once on startup ──────────────────────────────
model = joblib.load("models/tuned_xgboost_pipeline.pkl")

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


# ── Endpoints ────────────────────────────────────────────────
@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.post("/predict")
def predict(data: CustomerInput):
    df = pd.DataFrame([data.model_dump()])

    prediction  = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    if probability >= 0.70:
        risk = "high"
    elif probability >= 0.40:
        risk = "medium"
    else:
        risk = "low"

    return {
        "prediction":  prediction,
        "probability": round(probability, 4),
        "risk":        risk,
    }


# ── Dev runner ───────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
