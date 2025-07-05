import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="KHEHRAGSK AI Signal", layout="centered")

# --- Symbol List with OTC Pairs (You can expand more later)
SYMBOLS = {
    "EUR/USD (OTC)": "EURUSD=X",
    "GBP/USD (OTC)": "GBPUSD=X",
    "USD/JPY (OTC)": "USDJPY=X",
    "AUD/USD (OTC)": "AUDUSD=X",
    "USD/CHF (OTC)": "USDCHF=X",
    "NZD/USD (OTC)": "NZDUSD=X",
    "USD/CAD (OTC)": "USDCAD=X",
    "BTC/USD (OTC)": "BTC-USD",
    "ETH/USD (OTC)": "ETH-USD"
}

EXPIRIES = {"1m": 1, "3m": 3, "5m": 5}
LOOKBACK = 30

# --- Load & Prepare Data
def load_data(symbol):
    df = yf.download(symbol, period="2d", interval="1m").dropna()
    df["return"] = df["Close"].pct_change().fillna(0)
    df["ma5"] = df["Close"].rolling(5).mean().fillna(method="bfill")
    df["ma20"] = df["Close"].rolling(20).mean().fillna(method="bfill")
    return df

def prepare_data(df, expiry):
    X, y = [], []
    for i in range(LOOKBACK, len(df) - expiry):
        features = df.iloc[i - LOOKBACK:i][["return", "ma5", "ma20"]].values
        label = int(df["Close"].iloc[i + expiry] > df["Close"].iloc[i])
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

# --- Streamlit UI
st.title("📡 KHEHRAGSK Binary AI Signal")
pair = st.selectbox("Select OTC Pair", list(SYMBOLS.keys()))
expiry = st.selectbox("Expiry Time", list(EXPIRIES.keys()))

if st.button("📊 Generate Signal"):
    with st.spinner("⏳ Analyzing market using AI..."):
        try:
            df = load_data(SYMBOLS[pair])
            X, y = prepare_data(df, EXPIRIES[expiry])
            X_flat = X.reshape((X.shape[0], -1))

            model = RandomForestClassifier()
            model.fit(X_flat, y)
            pred = model.predict_proba([X_flat[-1]])[0][1]

            signal = "CALL 🔼" if pred > 0.5 else "PUT 🔽"
            confidence = round(pred * 100 if pred > 0.5 else (1 - pred) * 100, 2)

            st.success(f"""
                ✅ **Signal:** {signal}  
                📈 **Confidence:** {confidence}%  
                ⏰ **Expiry:** {expiry}
            """)
        except Exception as e:
            st.error(f"Error: {str(e)}")
