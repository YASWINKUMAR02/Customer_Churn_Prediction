# -*- coding: utf-8 -*-
"""
ChurnGuard AI · Sentinel Dark — Native Streamlit Edition
Self-contained, 100% cloud-deployment ready with integrated TreeSHAP explainability.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st

# ── 1. Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI · Sentinel Dark",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 2. Model & Explainer Loading (Cached) ─────────────────────────────
@st.cache_resource
def get_model_assets():
    model_path = os.path.join(os.path.dirname(__file__), "models", "tuned_xgboost_pipeline.pkl")
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
    return pipeline, preprocessor, xgb_model, explainer

try:
    pipeline, preprocessor, xgb_model, explainer = get_model_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_err = str(e)

# ── 3. Custom CSS: Sentinel Dark Full Design System ──────────────────
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
    /* ─── Global Variables & Reset ─── */
    :root {
        --bg-base: #0B0F14;
        --bg-panel: #141A22;
        --bg-sidebar: #0D1219;
        --bg-input: #111820;
        --border: #232B36;
        --border-subtle: #1A2330;
        --accent: #00D9A3;
        --accent-dim: rgba(0,217,163,0.12);
        --accent-glow: rgba(0,217,163,0.25);
        --risk-low: #22C55E;
        --risk-low-dim: rgba(34,197,94,0.12);
        --risk-med: #F59E0B;
        --risk-med-dim: rgba(245,158,11,0.12);
        --risk-high: #EF4444;
        --risk-high-dim: rgba(239,68,68,0.12);
        --text: #E8ECF1;
        --text-muted: #8A94A3;
        --text-dim: #4A5568;
        --purple: #A78BFA;
        --purple-dim: rgba(167,139,250,0.12);
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-base) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Hide standard Streamlit header & toolbar */
    #MainMenu, header, footer, [data-testid="stHeader"], [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1600px !important;
    }

    /* ─── Custom Header Bar ─── */
    .sentinel-header {
        background: var(--bg-sidebar);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    .sh-icon {
        width: 38px;
        height: 38px;
        background: var(--accent-dim);
        border: 1px solid rgba(0,217,163,0.3);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
    }
    .sh-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.1;
    }
    .sh-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        color: var(--text-muted);
        letter-spacing: 0.06em;
    }
    .sh-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--accent);
        background: var(--accent-dim);
        border: 1px solid rgba(0,217,163,0.25);
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.08em;
    }
    .sh-shap-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--purple);
        background: var(--purple-dim);
        border: 1px solid rgba(167,139,250,0.25);
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.08em;
        margin-left: auto;
    }

    /* ─── Form & Sidebar Cards ─── */
    .sentinel-card {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .card-teal { border-left: 3px solid var(--accent); }
    .card-indigo { border-left: 3px solid #818CF8; }
    .card-violet { border-left: 3px solid #A78BFA; }

    .card-head {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 0.9rem;
    }
    .c-icon {
        width: 30px;
        height: 30px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
    }
    .ci-teal { background: var(--accent-dim); border: 1px solid rgba(0,217,163,0.25); }
    .ci-indigo { background: rgba(129,140,248,0.12); border: 1px solid rgba(129,140,248,0.25); }
    .ci-violet { background: rgba(167,139,250,0.12); border: 1px solid rgba(167,139,250,0.25); }

    .c-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text);
        line-height: 1.1;
    }
    .c-desc {
        font-size: 0.7rem;
        color: var(--text-muted);
    }

    /* ─── Sidebar Stat Pills ─── */
    .stat-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--bg-input);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.45rem 0.65rem;
        margin-bottom: 0.4rem;
    }
    .stat-lbl { font-size: 0.72rem; color: var(--text-muted); }
    .stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--accent);
        background: var(--accent-dim);
        border: 1px solid rgba(0,217,163,0.2);
        padding: 2px 7px;
        border-radius: 4px;
    }

    /* ─── Results Components ─── */
    .res-grid {
        display: grid;
        grid-template-columns: 1.2fr 1fr 1fr;
        gap: 0.75rem;
        margin-bottom: 0.85rem;
    }
    .res-card {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }
    .res-churn {
        border-color: rgba(239,68,68,0.35);
        box-shadow: 0 0 0 1px rgba(239,68,68,0.1), inset 0 0 35px rgba(239,68,68,0.04);
    }
    .res-stay {
        border-color: rgba(34,197,94,0.35);
        box-shadow: 0 0 0 1px rgba(34,197,94,0.1), inset 0 0 35px rgba(34,197,94,0.04);
    }
    .res-eyebrow {
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .res-headline {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }
    .hl-churn { color: #FCA5A5; }
    .hl-stay { color: #86EFAC; }
    
    .prob-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.35rem 0 0.2rem;
        background: linear-gradient(135deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.4rem 0.85rem;
        border-radius: 50px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 0.35rem 0;
    }
    .rb-low { background: var(--risk-low-dim); border: 1px solid rgba(34,197,94,0.3); color: var(--risk-low); }
    .rb-med { background: var(--risk-med-dim); border: 1px solid rgba(245,158,11,0.3); color: var(--risk-med); }
    .rb-high { background: var(--risk-high-dim); border: 1px solid rgba(239,68,68,0.3); color: var(--risk-high); }

    /* ─── Risk Meter Gauge ─── */
    .meter-box {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.85rem;
    }
    .meter-track-wrap { position: relative; height: 9px; margin-top: 0.6rem; margin-bottom: 0.4rem; }
    .meter-track {
        position: absolute;
        inset: 0;
        border-radius: 5px;
        background: linear-gradient(90deg, #22C55E 0%, #22C55E 38%, #F59E0B 38%, #F59E0B 55%, #EF4444 55%, #EF4444 100%);
        opacity: 0.25;
    }
    .meter-fill {
        position: absolute;
        top: 0; left: 0; bottom: 0;
        border-radius: 5px;
        background: linear-gradient(90deg, #22C55E 0%, #F59E0B 50%, #EF4444 100%);
    }
    .meter-needle {
        position: absolute;
        top: -4px;
        width: 3px;
        height: 17px;
        background: #fff;
        border-radius: 2px;
        transform: translateX(-50%);
        box-shadow: 0 0 8px rgba(255,255,255,0.7);
    }
    .thresh-row {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }

    /* ─── SHAP Feature Attribution Items ─── */
    .shap-item {
        display: flex;
        flex-direction: column;
        gap: 3px;
        background: var(--bg-input);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 0.45rem 0.6rem;
        margin-bottom: 0.45rem;
    }
    .shap-row-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
    }
    .shap-val-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        padding: 1px 5px;
        border-radius: 3px;
    }
    .val-churn { color: #FCA5A5; background: rgba(239,68,68,0.12); }
    .val-stay { color: #6EE7B7; background: rgba(0,217,163,0.12); }

    .shap-track {
        position: relative;
        height: 5px;
        background: rgba(255,255,255,0.05);
        border-radius: 3px;
        overflow: hidden;
        margin-top: 2px;
    }
    .shap-fill-churn {
        position: absolute;
        top: 0; bottom: 0; right: 0;
        background: linear-gradient(90deg, #F87171, #EF4444);
        box-shadow: 0 0 6px rgba(239,68,68,0.3);
        border-radius: 3px;
    }
    .shap-fill-stay {
        position: absolute;
        top: 0; bottom: 0; left: 0;
        background: linear-gradient(90deg, #00D9A3, #38BDF8);
        box-shadow: 0 0 6px rgba(0,217,163,0.3);
        border-radius: 3px;
    }

    /* ─── Streamlit Widget Overrides ─── */
    .stSelectbox label, .stRadio label, .stNumberInput label, .stCheckbox label {
        color: var(--text-muted) !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    div[data-baseweb="select"] > div, input {
        background-color: var(--bg-input) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        border-radius: 6px !important;
    }
    
    /* Predict Button Glow */
    div.stButton > button {
        background: var(--accent) !important;
        color: #0B0F14 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 0 20px rgba(0,217,163,0.3) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 30px rgba(0,217,163,0.5) !important;
        transform: translateY(-1px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 4. Top Header Bar ────────────────────────────────────────────────
st.markdown(
    """
    <div class="sentinel-header">
        <div class="sh-icon">🛡️</div>
        <div>
            <div class="sh-title">ChurnGuard AI</div>
            <div class="sh-sub">POWERED BY XGBOOST</div>
        </div>
        <div style="width:1px; height:24px; background:#232B36; margin-left:0.5rem;"></div>
        <div class="sh-badge">SENTINEL DARK v2.0</div>
        <div class="sh-shap-badge">🧠 SHAP EXPLAINER ACTIVE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Helper for clean feature names ──────────────────────────────────
def clean_feat_name(raw: str) -> str:
    cleaned = raw.replace("cat__", "").replace("num__", "")
    if "_" in cleaned:
        p = cleaned.split("_", 1)
        return f"{p[0]}: {p[1]}"
    if cleaned == "tenure": return "Tenure Length"
    if cleaned == "MonthlyCharges": return "Monthly Charges"
    if cleaned == "TotalCharges": return "Total Charges"
    if cleaned == "SeniorCitizen": return "Senior Citizen"
    return cleaned

# ── 5. Main 3-Column Layout ──────────────────────────────────────────
col_sidebar, col_main, col_shap = st.columns([1, 2.7, 1.4])

# ── COLUMN 1: SIDEBAR ───────────────────────────────────────────────
with col_sidebar:
    st.markdown(
        """
        <div class="sentinel-card">
            <div style="font-family:'Space Grotesk'; font-size:0.7rem; font-weight:600; color:#8A94A3; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.75rem;">
                ℹ️ Model Info
            </div>
            <div class="stat-pill"><span class="stat-lbl">⚡ Algorithm</span><span class="stat-val">XGBoost</span></div>
            <div class="stat-pill"><span class="stat-lbl">📄 Input Features</span><span class="stat-val">19</span></div>
            <div class="stat-pill"><span class="stat-lbl">🎯 Hyperparams</span><span class="stat-val">Tuned</span></div>
            <div class="stat-pill"><span class="stat-lbl">🧠 Explainability</span><span class="stat-val" style="color:#A78BFA; background:rgba(167,139,250,0.12); border-color:rgba(167,139,250,0.2);">TreeSHAP</span></div>
        </div>

        <div class="sentinel-card">
            <div style="font-family:'Space Grotesk'; font-size:0.7rem; font-weight:600; color:#8A94A3; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.75rem;">
                📖 How to Use
            </div>
            <div style="font-size:0.74rem; color:#8A94A3; line-height:1.5;">
                <div style="margin-bottom:0.4rem;"><strong>1.</strong> Fill in <strong>Customer Profile</strong> demographics.</div>
                <div style="margin-bottom:0.4rem;"><strong>2.</strong> Choose <strong>connectivity</strong> & billing methods.</div>
                <div style="margin-bottom:0.4rem;"><strong>3.</strong> Toggle active <strong>add-on services</strong>.</div>
                <div><strong>4.</strong> Click <strong>Predict Churn</strong> to view results &amp; SHAP tree impacts.</div>
            </div>
        </div>

        <div style="text-align:center; font-family:'JetBrains Mono'; font-size:0.6rem; color:#4A5568; margin-top:1rem;">
            &copy; 2024 ChurnGuard AI &middot; v2.0<br>Model: tuned_xgboost_pipeline.pkl
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── COLUMN 2: MAIN FORM & RESULTS ────────────────────────────────────
with col_main:
    # CARD 1: Customer Profile
    st.markdown(
        """
        <div class="sentinel-card card-teal" style="margin-bottom:0.5rem;">
            <div class="card-head">
                <div class="c-icon ci-teal">👤</div>
                <div><div class="c-title">Customer Profile</div><div class="c-desc">Demographics and account details</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with p2:
        senior = st.checkbox("Senior Citizen")
    with p3:
        partner = st.checkbox("Partner")
    with p4:
        dependents = st.checkbox("Dependents")

    p5, p6, p7, p8 = st.columns(4)
    with p5:
        tenure = st.number_input("Tenure (months)", min_value=1, max_value=72, value=12)
    with p6:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with p7:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.25, max_value=118.75, value=70.00, step=0.01)
    with p8:
        total_charges = st.number_input("Total Charges ($)", min_value=18.80, max_value=8684.80, value=1000.00, step=0.01)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # CARD 2: Connectivity & Phone
    st.markdown(
        """
        <div class="sentinel-card card-indigo" style="margin-bottom:0.5rem;">
            <div class="card-head">
                <div class="c-icon ci-indigo">📡</div>
                <div><div class="c-title">Connectivity &amp; Phone</div><div class="c-desc">Phone, internet and billing preferences</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        phone_service = st.checkbox("Phone Service", value=True)
    with c2:
        if not phone_service:
            multi_lines = "No phone service"
            st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
        else:
            multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    with c3:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    c4, c5 = st.columns([1.6, 1])
    with c4:
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    with c5:
        paperless_billing = st.checkbox("Paperless Billing")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # CARD 3: Add-on Services
    st.markdown(
        """
        <div class="sentinel-card card-violet" style="margin-bottom:0.5rem;">
            <div class="card-head">
                <div class="c-icon ci-violet">🛠️</div>
                <div><div class="c-title">Add-on Services</div><div class="c-desc">Security, support &amp; streaming subscriptions</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def iv(checked):
        return "No internet service" if internet_service == "No" else ("Yes" if checked else "No")

    a1, a2, a3, a4, a5, a6 = st.columns(6)
    with a1: sec = st.checkbox("Security")
    with a2: bkp = st.checkbox("Backup")
    with a3: dvp = st.checkbox("Device")
    with a4: tsp = st.checkbox("Tech Supp")
    with a5: stv = st.checkbox("Stream TV")
    with a6: smv = st.checkbox("Movies")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("▶  Predict Churn Probability", use_container_width=True)

    # Prediction Computation
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior else 0,
        "Partner": "Yes" if partner else "No",
        "Dependents": "Yes" if dependents else "No",
        "tenure": int(tenure),
        "PhoneService": "Yes" if phone_service else "No",
        "MultipleLines": multi_lines,
        "InternetService": internet_service,
        "OnlineSecurity": iv(sec),
        "OnlineBackup": iv(bkp),
        "DeviceProtection": iv(dvp),
        "TechSupport": iv(tsp),
        "StreamingTV": iv(stv),
        "StreamingMovies": iv(smv),
        "Contract": contract,
        "PaperlessBilling": "Yes" if paperless_billing else "No",
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    df_in = pd.DataFrame([payload])
    
    if model_loaded:
        pred = int(pipeline.predict(df_in)[0])
        prob = float(pipeline.predict_proba(df_in)[0][1])
        pct = round(prob * 100, 1)
        risk = "high" if prob >= 0.70 else ("medium" if prob >= 0.40 else "low")
        
        # Calculate SHAP values
        transformed = preprocessor.transform(df_in)
        feat_names = preprocessor.get_feature_names_out()
        shap_res = explainer(transformed)
        shap_vals = shap_res.values[0]

        shap_items = []
        for fn, sv in zip(feat_names, shap_vals):
            val = float(sv)
            if abs(val) > 0.005:
                shap_items.append({
                    "name": clean_feat_name(fn),
                    "val": round(val, 3),
                    "abs": abs(val),
                    "is_churn": val > 0
                })
        shap_items = sorted(shap_items, key=lambda x: x["abs"], reverse=True)[:10]
    else:
        pred, prob, pct, risk, shap_items = 0, 0.0, 0.0, "low", []

    # Display Prediction Results
    if predict_btn:
        st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.75rem;">
                <div style="display:flex; align-items:center; gap:6px; font-family:'JetBrains Mono'; font-size:0.68rem; font-weight:600; color:#22C55E; background:rgba(34,197,94,0.12); border:1px solid rgba(34,197,94,0.25); padding:3px 10px; border-radius:20px;">
                    ● Analysis Complete
                </div>
                <div style="font-family:'Space Grotesk'; font-size:1.05rem; font-weight:700; color:#E8ECF1;">Prediction Result</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        card_cls = "res-churn" if pred == 1 else "res-stay"
        hl_cls = "hl-churn" if pred == 1 else "hl-stay"
        hl_txt = "⚠️ Will Churn" if pred == 1 else "✅ Will Stay"
        hl_sub = "Customer shows a high likelihood of leaving." if pred == 1 else "Customer is likely to remain subscribed."
        
        rb_cls = f"rb-{risk}"
        rb_label = {"low": "🟢 Low Churn Risk", "medium": "🟡 Medium Churn Risk", "high": "🔴 High Churn Risk"}[risk]

        st.markdown(
            f"""
            <div class="res-grid">
                <div class="res-card {card_cls}">
                    <div class="res-eyebrow">Prediction</div>
                    <div class="res-headline {hl_cls}">{hl_txt}</div>
                    <div style="font-size:0.72rem; color:#8A94A3;">{hl_sub}</div>
                </div>
                <div class="res-card">
                    <div class="res-eyebrow">Churn Probability</div>
                    <div class="prob-num">{pct}%</div>
                    <div style="font-size:0.72rem; color:#8A94A3;">Model confidence score</div>
                </div>
                <div class="res-card" style="display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
                    <div class="res-eyebrow" style="width:100%;">Risk Level</div>
                    <div class="risk-badge {rb_cls}">{rb_label}</div>
                    <div style="font-size:0.72rem; color:#8A94A3;">Based on churn threshold</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Risk Meter
        st.markdown(
            f"""
            <div class="meter-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:'Space Grotesk'; font-size:0.75rem; font-weight:600; color:#8A94A3; text-transform:uppercase;">Churn Risk Meter</span>
                    <span style="font-family:'JetBrains Mono'; font-size:0.8rem; font-weight:600; color:#00D9A3;">{pct}% Probability</span>
                </div>
                <div class="meter-track-wrap">
                    <div class="meter-track"></div>
                    <div class="meter-fill" style="width:{pct}%;"></div>
                    <div class="meter-needle" style="left:{pct}%;"></div>
                </div>
                <div class="thresh-row">
                    <span style="color:#22C55E;">0% — Low</span>
                    <span style="color:#F59E0B;">40% — Medium</span>
                    <span style="color:#EF4444;">70% — High</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── COLUMN 3: RIGHT SHAP EXPLAINER ──────────────────────────────────
with col_shap:
    st.markdown(
        """
        <div class="sentinel-card">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem;">
                <div style="display:flex; align-items:center; gap:7px;">
                    <span style="font-size:1.1rem;">🧠</span>
                    <span style="font-family:'Space Grotesk'; font-size:0.85rem; font-weight:700; color:#E8ECF1;">SHAP Explainer</span>
                </div>
                <span style="font-family:'JetBrains Mono'; font-size:0.6rem; font-weight:600; color:#A78BFA; background:rgba(167,139,250,0.12); border:1px solid rgba(167,139,250,0.25); padding:2px 6px; border-radius:4px;">LOCAL IMPACT</span>
            </div>
            <p style="font-size:0.7rem; color:#8A94A3; line-height:1.35; margin-bottom:0.75rem;">
                Quantifies how each feature pushes the prediction toward <strong>Churn</strong> or <strong>Retention</strong> using TreeSHAP values.
            </p>
            <div style="display:flex; justify-content:space-between; background:#111820; border:1px solid #232B36; border-radius:6px; padding:0.45rem 0.65rem; font-size:0.65rem; font-family:'JetBrains Mono'; margin-bottom:0.85rem;">
                <div style="display:flex; align-items:center; gap:5px;"><span style="width:7px; height:7px; border-radius:50%; background:#EF4444;"></span><span style="color:#FCA5A5;">Pushes Churn (+)</span></div>
                <div style="display:flex; align-items:center; gap:5px;"><span style="width:7px; height:7px; border-radius:50%; background:#00D9A3;"></span><span style="color:#6EE7B7;">Pushes Retention (-)</span></div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if not predict_btn:
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:2.5rem 1rem; border:1px dashed #232B36; border-radius:8px; background:rgba(17,24,32,0.4);">
                <div style="font-size:1.8rem; margin-bottom:0.6rem; opacity:0.7;">📊</div>
                <div style="font-family:'Space Grotesk'; font-size:0.8rem; font-weight:600; color:#E8ECF1; margin-bottom:0.25rem;">Awaiting Prediction</div>
                <div style="font-size:0.68rem; color:#4A5568; max-width:200px;">Click "Predict Churn Probability" to compute local SHAP feature attributions.</div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        max_abs = max([x["abs"] for x in shap_items], default=0.01)
        bars_html = ""
        for item in shap_items:
            pct_w = min(int((item["abs"] / max_abs) * 100), 100)
            sign = "+" if item["val"] > 0 else ""
            badge_cls = "val-churn" if item["is_churn"] else "val-stay"
            fill_cls = "shap-fill-churn" if item["is_churn"] else "shap-fill-stay"
            
            bars_html += f"""
            <div class="shap-item">
                <div class="shap-row-top">
                    <span style="font-weight:500; color:#E8ECF1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:200px;" title="{item['name']}">{item['name']}</span>
                    <span class="shap-val-badge {badge_cls}">{sign}{item['val']}</span>
                </div>
                <div class="shap-track">
                    <div class="{fill_cls}" style="width:{pct_w}%;"></div>
                </div>
            </div>
            """

        top_churn = next((x for x in shap_items if x["is_churn"]), None)
        top_stay = next((x for x in shap_items if not x["is_churn"]), None)
        insight_msg = ""
        if top_churn:
            insight_msg += f"Primary risk driver is <strong style='color:#A78BFA;'>{top_churn['name']}</strong> (+{top_churn['val']} SHAP). "
        if top_stay:
            insight_msg += f"Strongest retention anchor is <strong style='color:#00D9A3;'>{top_stay['name']}</strong> ({top_stay['val']} SHAP)."

        st.markdown(bars_html + "</div>", unsafe_allow_html=True)
        
        st.markdown(
            f"""
            <div class="sentinel-card" style="margin-top:0.75rem;">
                <div style="font-family:'Space Grotesk'; font-size:0.7rem; font-weight:600; color:#8A94A3; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">
                    💡 Attribution Summary
                </div>
                <div style="background:#111820; border-left:3px solid #A78BFA; border-radius:6px; padding:0.6rem 0.75rem; font-size:0.7rem; line-height:1.4; color:#8A94A3;">
                    {insight_msg}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
