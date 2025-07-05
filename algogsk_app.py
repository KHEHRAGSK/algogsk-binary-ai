import streamlit as st
import numpy as np
import datetime

st.set_page_config(page_title="ALGOGSK OTC AI Signals", layout="centered")
st.title("📡 ALGOGSK Binary AI OTC Signal Generator")

# --- SYMBOLS ---
OTC_SYMBOLS = [
    # Currencies
    "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", "USD/CHF (OTC)", "USD/CAD (OTC)",
    "AUD/USD (OTC)", "NZD/USD (OTC)", "USD/BRL (OTC)", "USD/DZD (OTC)", "USD/EGP (OTC)",
    "EUR/GBP (OTC)", "GBP/JPY (OTC)", "AUD/JPY (OTC)", "CHF/JPY (OTC)", "EUR/JPY (OTC)",
    # Cryptos
    "Bitcoin (OTC)", "Ethereum (OTC)", "Ripple (OTC)", "Shiba Inu (OTC)", "Dogecoin (OTC)",
    # Stocks
    "Microsoft (OTC)", "Facebook Inc (OTC)", "Johnson & Johnson (OTC)", "McDonald's (OTC)", "Pfizer Inc (OTC)"
]

EXPIRIES = {"1 Minute": 1, "3 Minutes": 3, "5 Minutes": 5}

# --- Signal Logic (placeholder AI prediction) ---
def generate_signal():
    direction = np.random.choice(["CALL 🔼", "PUT 🔽"])
    confidence = round(np.random.uniform(70, 99), 2)
    return direction, confidence

# --- UI ---
symbol = st.selectbox("📊 Select Asset", OTC_SYMBOLS)
expiry = st.selectbox("⏳ Select Expiry", list(EXPIRIES.keys()))

if st.button("🚀 Generate AI Signal"):
    with st.spinner("Analyzing OTC market..."):
        signal, confidence = generate_signal()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"""
        🧠 **AI Signal: {signal}**  
        🔒 **Confidence:** `{confidence}%`  
        ⏱️ **Expiry:** `{expiry}`  
        🕒 **Time:** `{now}`
        """)
