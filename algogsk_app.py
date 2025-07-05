import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

st.set_page_config(page_title="💹 ALGOGSK Binary AI Signal", layout="centered")
st.title("💹 ALGOGSK Binary AI Signal Generator")

# Supported expiry times
EXPIRIES = {"1m": 1, "3m": 3, "5m": 5}

# Full OTC Pairs List (add/remove as needed)
SYMBOLS = {
    "EUR/USD (OTC)": "EURUSD=X",
    "GBP/USD (OTC)": "GBPUSD=X",
    "USD/JPY (OTC)": "USDJPY=X",
    "AUD/USD (OTC)": "AUDUSD=X",
    "USD/CHF (OTC)": "USDCHF=X",
    "NZD/USD (OTC)": "NZDUSD=X",
    "USD/CAD (OTC)": "USDCAD=X",
    "EUR/GBP (OTC)": "EURGBP=X",
    "EUR/JPY (OTC)": "EURJPY=X",
    "GBP/JPY (OTC)": "GBPJPY=X",
    "AUD/JPY (OTC)": "AUDJPY=X",
    "CHF/JPY (OTC)": "CHFJPY=X"
}

# Load price data
@st.cache_data
def load_data(symbol, period="2d", interval="1m"):
    df = yf.download(symbol, period=period, interval=interval).dropna()
    return df

# Create features
def make_features(df):
    df["return"] = df["Close"].pct_change()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["RSI"] = 100 - 100 / (1 + df["return"].rolling(14).mean() / df["return"].rolling(14).std())
    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["Close"].rolling(20).mean() + 2 * df["Close"].rolling(20).std()
    df["Lower"] = df["Close"].rolling(20).mean() - 2 * df["Close"].rolling(20).std()
    df = df.dropna()
    return df

# Train and predict
def predict_signal(df, expiry_candles):
    X, y = [], []
    for i in range(60, len(df) - expiry_candles):
        features = df[["return", "EMA20", "RSI", "MACD", "Signal", "Upper", "Lower"]].iloc[i - 60:i].values
        target = int(df["Close"].iloc[i + expiry_candles] > df["Close"].iloc[i])
        X.append(features)
        y.append(target)
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        return None, "Insufficient data"

    X = X.reshape(X.shape[0], -1)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    last_features = df[["return", "EMA20", "RSI", "MACD", "Signal", "Upper", "Lower"]].iloc[-60:].values.flatten().reshape(1, -1)
    pred = model.predict_proba(last_features)[0][1]
    signal = "CALL 🔼" if pred > 0.5 else "PUT 🔽"
    confidence = round(pred * 100 if pred > 0.5 else (1 - pred) * 100, 2)
    return signal, confidence

# UI
pair = st.selectbox("Select Currency Pair", list(SYMBOLS.keys()))
expiry = st.selectbox("Select Expiry", list(EXPIRIES.keys()))
if st.button("Generate Signal"):
    with st.spinner("Fetching market data & generating signal..."):
        df = load_data(SYMBOLS[pair])
        df = make_features(df)
        signal, conf = predict_signal(df, EXPIRIES[expiry])
        if signal:
            st.success(f"**Signal: {signal}**\n\nConfidence: **{conf}%**\n\nExpiry: **{expiry}**")
        else:
            st.error(conf)
