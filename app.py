# -*- coding: utf-8 -*-
"""
ChurnGuard AI - Sentinel Dark - Streamlit Edition
Run with:  streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ChurnGuard AI - Sentinel Dark",
    page_icon="\U0001f6e1",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background-color:#0B0F14!important;color:#E8ECF1!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:0!important;max-width:100%!important;}

/* Header */
.sg-header{background:#0D1219;border-bottom:1px solid #232B36;padding:0 2rem;height:64px;display:flex;align-items:center;gap:1rem;position:relative;overflow:hidden;}
.sg-header-icon{width:38px;height:38px;background:rgba(0,217,163,0.12);border:1px solid rgba(0,217,163,0.3);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;}
.sg-header-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.15rem;color:#E8ECF1;letter-spacing:-0.01em;line-height:1;}
.sg-header-sub{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#8A94A3;letter-spacing:0.06em;}
.sg-header-divider{width:1px;height:28px;background:#232B36;margin-left:0.5rem;}
.sg-header-badge{font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:500;color:#00D9A3;background:rgba(0,217,163,0.12);border:1px solid rgba(0,217,163,0.25);padding:3px 10px;border-radius:20px;letter-spacing:0.08em;}

/* Sidebar */
[data-testid="stSidebar"]{background:#0D1219!important;border-right:1px solid #232B36!important;}
[data-testid="stSidebar"] *{color:#E8ECF1!important;}
.sidebar-card{background:#141A22;border:1px solid #1A2330;border-radius:14px;padding:1rem;margin-bottom:1rem;}
.sidebar-card-title{font-family:'Space Grotesk',sans-serif;font-size:0.72rem;font-weight:600;color:#8A94A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.85rem;}
.stat-pill{display:flex;align-items:center;justify-content:space-between;background:#111820;border:1px solid #232B36;border-radius:6px;padding:0.6rem 0.75rem;margin-bottom:0.5rem;}
.stat-pill-label{font-size:0.75rem;color:#8A94A3;}
.stat-pill-value{font-family:'JetBrains Mono',monospace;font-size:0.72rem;font-weight:600;color:#00D9A3;background:rgba(0,217,163,0.12);border:1px solid rgba(0,217,163,0.2);padding:2px 8px;border-radius:4px;}
.step-item{display:flex;gap:10px;font-size:0.78rem;color:#8A94A3;line-height:1.45;margin-bottom:0.6rem;}
.step-num{flex-shrink:0;width:20px;height:20px;background:#111820;border:1px solid #232B36;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:600;color:#00D9A3;}
.sidebar-footer{font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#4A5568;text-align:center;letter-spacing:0.04em;padding-top:1rem;border-top:1px solid #1A2330;}

/* Form cards */
.form-card{background:#141A22;border:1px solid #232B36;border-radius:14px;padding:1.2rem 1.5rem 0.8rem;margin-bottom:0.75rem;}
.form-card-teal{border-left:3px solid #00D9A3;}
.form-card-indigo{border-left:3px solid #818CF8;}
.form-card-violet{border-left:3px solid #A78BFA;}
.card-header{display:flex;align-items:center;gap:10px;}
.card-icon{width:34px;height:34px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:1rem;}
.ci-teal{background:rgba(0,217,163,0.12);border:1px solid rgba(0,217,163,0.2);}
.ci-indigo{background:rgba(129,140,248,0.12);border:1px solid rgba(129,140,248,0.25);}
.ci-violet{background:rgba(167,139,250,0.12);border:1px solid rgba(167,139,250,0.25);}
.card-title{font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:600;color:#E8ECF1;}
.card-desc{font-size:0.75rem;color:#8A94A3;}

/* Widget overrides */
[data-testid="stSelectbox"]>div>div,[data-testid="stNumberInput"] input{background:#111820!important;border:1px solid #232B36!important;border-radius:6px!important;color:#E8ECF1!important;}
[data-testid="stSelectbox"]>label,[data-testid="stNumberInput"]>label,[data-testid="stCheckbox"]>label,[data-testid="stRadio"]>label{font-size:0.7rem!important;font-weight:600!important;color:#8A94A3!important;letter-spacing:0.07em!important;text-transform:uppercase!important;}
div[data-testid="stButton"]>button{width:100%;padding:0.9rem 2rem;background:#00D9A3!important;color:#0B0F14!important;border:none!important;border-radius:14px!important;font-family:'Space Grotesk',sans-serif!important;font-size:0.95rem!important;font-weight:700!important;letter-spacing:0.03em!important;}
div[data-testid="stButton"]>button:hover{box-shadow:0 0 28px 4px rgba(0,217,163,0.25),0 4px 16px rgba(0,217,163,0.3)!important;transform:translateY(-1px)!important;}

/* Result cards */
.result-grid{display:grid;grid-template-columns:1.3fr 1fr 1fr;gap:1rem;margin-bottom:1rem;}
.result-card{background:#141A22;border:1px solid #232B36;border-radius:14px;padding:1.25rem 1.4rem;}
.churn-card{border-color:rgba(239,68,68,0.3)!important;box-shadow:0 0 0 1px rgba(239,68,68,0.1),inset 0 0 40px rgba(239,68,68,0.04)!important;}
.stay-card{border-color:rgba(34,197,94,0.3)!important;box-shadow:0 0 0 1px rgba(34,197,94,0.1),inset 0 0 40px rgba(34,197,94,0.04)!important;}
.eyebrow{font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#8A94A3;margin-bottom:0.5rem;}
.rh{font-family:'Space Grotesk',sans-serif;font-size:1.25rem;font-weight:700;line-height:1.2;margin-bottom:0.4rem;}
.rh-churn{color:#FCA5A5;}.rh-stay{color:#86EFAC;}
.rsub{font-size:0.78rem;color:#8A94A3;}
.prob-num{font-family:'JetBrains Mono',monospace;font-size:2.6rem;font-weight:700;line-height:1;margin:0.5rem 0 0.25rem;background:linear-gradient(135deg,#a78bfa,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.rbadge{display:inline-flex;align-items:center;gap:7px;padding:0.5rem 1rem;border-radius:50px;font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:600;margin:0.5rem 0;letter-spacing:0.02em;}
.bl{background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.3);color:#22C55E;}
.bm{background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.3);color:#F59E0B;}
.bh{background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);color:#EF4444;}

/* Risk meter */
.meter-card{background:#141A22;border:1px solid #232B36;border-radius:14px;padding:1.25rem 1.5rem 1.4rem;margin-bottom:1rem;}
.m-title{font-family:'Space Grotesk',sans-serif;font-size:0.8rem;font-weight:600;color:#8A94A3;letter-spacing:0.05em;text-transform:uppercase;}
.m-pct{font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:600;color:#00D9A3;}
.m-outer{position:relative;height:10px;border-radius:5px;overflow:visible;margin:0.85rem 0 0.5rem;}
.m-bg{position:absolute;inset:0;border-radius:5px;background:linear-gradient(90deg,#22C55E 0%,#22C55E 38%,#F59E0B 38%,#F59E0B 55%,#EF4444 55%,#EF4444 100%);opacity:0.25;}
.m-fill{position:absolute;top:0;left:0;bottom:0;border-radius:5px;background:linear-gradient(90deg,#22C55E 0%,#F59E0B 50%,#EF4444 100%);}
.m-thresh{display:flex;justify-content:space-between;margin-top:0.4rem;}
.tl{font-family:'JetBrains Mono',monospace;font-size:0.63rem;font-weight:500;color:#22C55E;}
.tm{font-family:'JetBrains Mono',monospace;font-size:0.63rem;font-weight:500;color:#F59E0B;}
.th{font-family:'JetBrains Mono',monospace;font-size:0.63rem;font-weight:500;color:#EF4444;}

/* Indicators */
.ind-card{background:#141A22;border:1px solid #232B36;border-radius:14px;padding:1.25rem 1.5rem;}
.ind-title{font-family:'Space Grotesk',sans-serif;font-size:0.8rem;font-weight:600;color:#8A94A3;letter-spacing:0.07em;text-transform:uppercase;margin-bottom:1rem;}
.ind-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.65rem;}
.ind-chip{display:flex;align-items:flex-start;gap:8px;padding:0.7rem 0.85rem;border-radius:6px;border:1px solid transparent;}
.good{background:rgba(34,197,94,0.12);border-color:rgba(34,197,94,0.2)!important;}
.warn{background:rgba(245,158,11,0.12);border-color:rgba(245,158,11,0.2)!important;}
.cl{font-size:0.78rem;font-weight:500;color:#E8ECF1;}
.cs{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#8A94A3;}

/* Results header */
.res-hdr{display:flex;align-items:center;gap:10px;margin-bottom:1.25rem;}
.sl{display:flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:600;color:#22C55E;letter-spacing:0.1em;text-transform:uppercase;background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.25);padding:4px 12px;border-radius:20px;}
.pd{display:inline-block;width:7px;height:7px;background:#22C55E;border-radius:50%;}
.rt{font-family:'Space Grotesk',sans-serif;font-size:1.15rem;font-weight:700;color:#E8ECF1;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("models/tuned_xgboost_pipeline.pkl")


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ── Header ─────────────────────────────────────────────────────
st.markdown(
    '<div class="sg-header">'
    '<div class="sg-header-icon">\U0001f6e1\ufe0f</div>'
    '<div><div class="sg-header-title">ChurnGuard AI</div>'
    '<div class="sg-header-sub">POWERED BY XGBOOST</div></div>'
    '<div class="sg-header-divider"></div>'
    '<span class="sg-header-badge">SENTINEL DARK v2.0</span></div>',
    unsafe_allow_html=True,
)

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-card">'
        '<div class="sidebar-card-title">\u2139 Model Info</div>'
        '<div class="stat-pill"><span class="stat-pill-label">\u26a1 Algorithm</span><span class="stat-pill-value">XGBoost</span></div>'
        '<div class="stat-pill"><span class="stat-pill-label">\U0001f4cb Input Features</span><span class="stat-pill-value">19</span></div>'
        '<div class="stat-pill"><span class="stat-pill-label">\U0001f3af Hyperparams</span><span class="stat-pill-value">Tuned</span></div>'
        '</div>'
        '<div class="sidebar-card">'
        '<div class="sidebar-card-title">\U0001f4da How to Use</div>'
        '<div class="step-item"><span class="step-num">1</span><span>Fill in the <strong style="color:#00D9A3">Customer Profile</strong>.</span></div>'
        '<div class="step-item"><span class="step-num">2</span><span>Select <strong style="color:#00D9A3">connectivity</strong> options.</span></div>'
        '<div class="step-item"><span class="step-num">3</span><span>Toggle <strong style="color:#00D9A3">add-on services</strong>.</span></div>'
        '<div class="step-item"><span class="step-num">4</span><span>Click <strong style="color:#00D9A3">Predict Churn</strong>.</span></div>'
        '</div>'
        '<div class="sidebar-footer">&copy; 2024 ChurnGuard AI &middot; v2.0<br>Model: tuned_xgboost_pipeline.pkl</div>',
        unsafe_allow_html=True,
    )

# ── Card 1: Customer Profile ────────────────────────────────────
st.markdown(
    '<div class="form-card form-card-teal"><div class="card-header">'
    '<div class="card-icon ci-teal">\U0001f464</div>'
    '<div><div class="card-title">Customer Profile</div>'
    '<div class="card-desc">Demographics and account details</div></div></div></div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
with c2:
    senior = st.checkbox("Senior Citizen")
with c3:
    partner = st.checkbox("Partner")
with c4:
    dependents = st.checkbox("Dependents")
with c5:
    tenure = st.number_input("Tenure (months)", min_value=1, max_value=72, value=12)
with c6:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

col_m, col_t = st.columns(2)
with col_m:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.25, max_value=118.75, value=70.00, step=0.01, format="%.2f")
with col_t:
    total_charges = st.number_input("Total Charges ($)", min_value=18.80, max_value=8684.80, value=1000.00, step=0.01, format="%.2f")

# ── Card 2: Connectivity & Phone ────────────────────────────────
st.markdown(
    '<div class="form-card form-card-indigo"><div class="card-header">'
    '<div class="card-icon ci-indigo">\U0001f4e1</div>'
    '<div><div class="card-title">Connectivity &amp; Phone</div>'
    '<div class="card-desc">Phone, internet and billing preferences</div></div></div></div>',
    unsafe_allow_html=True,
)

cc1, cc2, cc3, cc4, cc5 = st.columns(5)
with cc1:
    phone_service = st.checkbox("Phone Service", value=True)
with cc2:
    if not phone_service:
        multi_lines = "No phone service"
        st.selectbox("Multiple Lines", ["No phone service"], disabled=True, key="ml_dis")
    else:
        multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
with cc3:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
with cc4:
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
with cc5:
    paperless_billing = st.checkbox("Paperless Billing")

# ── Card 3: Add-on Services ─────────────────────────────────────
st.markdown(
    '<div class="form-card form-card-violet"><div class="card-header">'
    '<div class="card-icon ci-violet">\U0001f6e0</div>'
    '<div><div class="card-title">Add-on Services</div>'
    '<div class="card-desc">Security, support &amp; streaming subscriptions</div></div></div></div>',
    unsafe_allow_html=True,
)


def iv(checked):
    return "No internet service" if internet_service == "No" else ("Yes" if checked else "No")


a1, a2, a3, a4, a5, a6 = st.columns(6)
with a1:
    sec = st.checkbox("Online Security")
with a2:
    bkp = st.checkbox("Online Backup")
with a3:
    dvp = st.checkbox("Device Protection")
with a4:
    tsp = st.checkbox("Tech Support")
with a5:
    stv = st.checkbox("Streaming TV")
with a6:
    smv = st.checkbox("Streaming Movies")

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
predict_clicked = st.button("\u25b6  Predict Churn Probability", use_container_width=True)
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Results ─────────────────────────────────────────────────────
if predict_clicked:
    if not model_loaded:
        st.error(f"Could not load model: {model_error}")
    else:
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

        df = pd.DataFrame([payload])
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
        pct = round(probability * 100, 1)
        risk = "high" if probability >= 0.70 else ("medium" if probability >= 0.40 else "low")

        pc  = "churn-card" if prediction == 1 else "stay-card"
        phc = "rh-churn"   if prediction == 1 else "rh-stay"
        pt  = "\u26a0\ufe0f Will Churn" if prediction == 1 else "\u2705 Will Stay"
        ps  = "Customer shows a high likelihood of leaving." if prediction == 1 else "Customer is likely to remain subscribed."
        bc  = {"low": "bl", "medium": "bm", "high": "bh"}[risk]
        bl  = {
            "low":    "\U0001f7e2 Low Churn Risk",
            "medium": "\U0001f7e1 Medium Churn Risk",
            "high":   "\U0001f534 High Churn Risk",
        }[risk]

        st.markdown(
            f"""
            <div class="res-hdr">
              <div class="sl"><span class="pd"></span>&nbsp;Analysis Complete</div>
              <span class="rt">Prediction Result</span>
            </div>
            <div class="result-grid">
              <div class="result-card {pc}">
                <div class="eyebrow">Prediction</div>
                <div class="rh {phc}">{pt}</div>
                <div class="rsub">{ps}</div>
              </div>
              <div class="result-card">
                <div class="eyebrow">Churn Probability</div>
                <div class="prob-num">{pct}%</div>
                <div class="rsub">Model confidence score</div>
              </div>
              <div class="result-card" style="display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;">
                <div class="eyebrow" style="width:100%;">Risk Level</div>
                <span class="rbadge {bc}">{bl}</span>
                <div class="rsub">Based on churn probability threshold</div>
              </div>
            </div>
            <div class="meter-card">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.85rem;">
                <span class="m-title">Churn Risk Meter</span>
                <span class="m-pct">{pct}% Probability</span>
              </div>
              <div class="m-outer">
                <div class="m-bg"></div>
                <div class="m-fill" style="width:{pct}%;"></div>
              </div>
              <div class="m-thresh">
                <span class="tl">0% -- Low</span>
                <span class="tm">40% -- Medium</span>
                <span class="th">70% -- High</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def chip(good, lg, lw, sub):
            ic  = "\u2705" if good else "\u26a0\ufe0f"
            lbl = lg if good else lw
            cls = "good" if good else "warn"
            return (
                f'<div class="ind-chip {cls}">'
                f'<span style="font-size:0.9rem;flex-shrink:0;">{ic}</span>'
                f'<div><div class="cl">{lbl}</div><div class="cs">{sub}</div></div>'
                f'</div>'
            )

        chips = (
            chip(tenure >= 12, "Established customer", "Short tenure - higher risk", f"Tenure: {int(tenure)} months")
            + chip(contract != "Month-to-month", "Long-term contract", "Month-to-month contract", f"Contract: {contract}")
            + chip(monthly_charges <= 80, "Moderate monthly charges", "High monthly charges", f"Monthly: ${monthly_charges:.2f}")
            + chip(internet_service != "Fiber optic", "Standard internet plan", "Fiber optic - often higher churn", f"Internet: {internet_service}")
        )

        st.markdown(
            f'<div class="ind-card"><div class="ind-title">\U0001f4ca Key Risk Indicators</div>'
            f'<div class="ind-grid">{chips}</div></div>',
            unsafe_allow_html=True,
        )
