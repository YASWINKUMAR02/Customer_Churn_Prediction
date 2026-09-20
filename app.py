# -*- coding: utf-8 -*-
"""
ChurnGuard AI — Sentinel Dark
Runs FastAPI backend automatically in background if needed and embeds the exact index.html UI
with 100% pixel-perfect fidelity.
"""

import os
import sys
import time
import socket
import threading
import streamlit as st
import streamlit.components.v1 as components

# ── 1. Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI · Sentinel Dark",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 2. Ensure FastAPI backend is active on port 8000 ──────────────
def is_port_in_use(port: int = 8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def run_fastapi():
    try:
        import uvicorn
        from api import app as fastapi_app
        config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        print(f"FastAPI background server exception: {e}")

if not is_port_in_use(8000):
    t = threading.Thread(target=run_fastapi, daemon=True)
    t.start()
    time.sleep(0.8)

# ── 3. Seamless Streamlit Fullscreen Styling ────────────────────────
st.markdown(
    """
    <style>
    /* Hide Streamlit top header, menu, and status decorations */
    #MainMenu, header, footer, [data-testid="stHeader"], [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Remove all container padding for edge-to-edge dark canvas */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        max-width: 100% !important;
    }
    
    .stApp {
        background-color: #0B0F14 !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }

    iframe {
        width: 100% !important;
        border: none !important;
        min-height: 100vh !important;
        background: #0B0F14 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 4. Load and Render the EXACT index.html ─────────────────────────
html_path = os.path.join(os.path.dirname(__file__), "index.html")
if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_code = f.read()
    components.html(html_code, height=1300, scrolling=True)
else:
    st.error("index.html not found in project directory.")
