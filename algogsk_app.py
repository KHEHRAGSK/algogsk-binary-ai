import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ALGOGSK Binary AI Signal", layout="centered")
st.title("📡 ALGOGSK Binary AI Signal Generator")

# --- Config
LOOKBACK = 60
EXPIRIES = {"1m": 1, "3m": 3, "5m": 5}
SYMBOLS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X", "USD/CAD": "USDCAD=X", "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X", "AUD/JPY": "AUDJPY=X", "CHF/JPY": "CHFJPY=X"
}

# --- Functions
def load_data(symbol, period="2d", interval="1m"):
    try:
        df = yf.download(symbol, period=period, interval=interval).dropna()
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def make_features(df):
    df["return"] = df["Close"].pct_change().fillna(0)
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["rsi"] = (100 - (100 / (1 + df["return"].rolling(14).mean() / df["return"].rolling(14).std()))).fillna(50)
    df = df.dropna()
    return df

def prepare_data(df, expiry_candles):
    X, y = [], []
    for i in range(LOOKBACK, len(df) - expiry_candles):
        X.append(df.iloc[i - LOOKBACK:i][["return", "ma5", "ma20", "rsi"]].values.flatten())
        future_move = df["Close"].iloc[i + expiry_candles] > df["Close"].iloc[i]
        y.append(int(future_move))
    return np.array(X), np.array(y)

def predict_signal(symbol, expiry_str):
    df = load_data(symbol)
    if df is None or len(df) < 100:
        return None, "Not enough data"
    df = make_features(df)
    X, y = prepare_data(df, EXPIRIES[expiry_str])
    if len(X) < 10:
        return None, "Insufficient training data"
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)
    prob = model.predict_proba(X[-1].reshape(1, -1))[0][1]
    signal = "CALL 🔼" if prob > 0.5 else "PUT 🔽"
    confidence = round(prob * 100 if prob > 0.5 else (1 - prob) * 100, 2)
    return signal, confidence

# --- UI
pair = st.selectbox("Select Currency Pair", list(SYMBOLS.keys()))
expiry = st.selectbox("Select Expiry", list(EXPIRIES.keys()))
if st.button("🧠 Generate AI Signal"):
    with st.spinner("Analyzing with AI..."):
        signal, result = predict_signal(SYMBOLS[pair], expiry)
        if signal is None:
            st.error(result)
        else:
            st.success(f"**Signal: {signal}**\n\n**Confidence:** `{result}%`\n\n**Expiry:** `{expiry}`")
