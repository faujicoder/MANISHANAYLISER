import os
import time
import math
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go

# ============== Helpers & Config ==============
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

@st.cache_data(show_spinner=False)
def get_api_key():
    try:
        return st.secrets["ALPHA_VANTAGE_API_KEY"]
    except Exception:
        return os.getenv("ALPHA_VANTAGE_API_KEY")

@st.cache_data(ttl=600, show_spinner=True)
def fetch_alpha_vantage_daily(ticker: str, full: bool = False) -> pd.DataFrame:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Alpha Vantage API key missing. Add it in Secrets.")
    function = "TIME_SERIES_DAILY_ADJUSTED"
    outputsize = "full" if full else "compact"
    url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&outputsize={outputsize}&apikey={api_key}"
    r = requests.get(url, timeout=30)
    payload = r.json()
    if "Note" in payload:
        raise RuntimeError("API rate limit hit. Try again in a minute.")
    if "Error Message" in payload:
        raise RuntimeError(payload["Error Message"])  
    ts = payload.get("Time Series (Daily)")
    if not ts:
        raise RuntimeError("No data returned for this ticker. Check symbol.")

    df = (
        pd.DataFrame(ts).T
        .rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
        })
    )
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index = pd.to_datetime(df.index)
    return df.sort_index().dropna()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["ret"] = X["close"].pct_change()
    X["hl_pct"] = (X["high"] - X["low"]) / X["close"].replace(0, np.nan)
    X["vol_chg"] = X["volume"].pct_change()
    for w in [5, 10, 20, 50]:
        X[f"ma_{w}"] = X["close"].rolling(w).mean()
        X[f"ma_ratio_{w}"] = X["close"] / (X[f"ma_{w}"] + 1e-9)
    X["rsi_14"] = rsi(X["close"], 14)
    for lag in [1, 2, 3, 5]:
        X[f"close_lag_{lag}"] = X["close"].shift(lag)
        X[f"vol_lag_{lag}"] = X["volume"].shift(lag)
    X["dow"] = X.index.dayofweek
    X["month"] = X.index.month
    X["target"] = X["close"].shift(-1)
    return X.dropna()

def time_series_train_test_split(X: pd.DataFrame, test_size: float = 0.2):
    n_test = max(1, int(len(X) * test_size))
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    features = [c for c in X.columns if c != "target"]
    return X_train[features], X_test[features], X_train["target"], X_test["target"], features

@st.cache_resource(show_spinner=False)
def build_model():
    return RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )

def forecast_next_days(model, last_row: pd.Series, days: int = 7) -> pd.DataFrame:
    preds, row = [], last_row.copy()
    for i in range(days):
        yhat = model.predict(row.values.reshape(1, -1))[0]
        preds.append({"day": i + 1, "pred_close": float(yhat)})
        prev_close = row.get("close", np.nan)
        row["close"] = yhat
        for lag in [5, 3, 2, 1]:
            if f"close_lag_{lag}" in row.index:
                row[f"close_lag_{lag}"] = prev_close if lag == 1 else row.get(f"close_lag_{lag-1}", prev_close)
        for w in [5, 10, 20, 50]:
            ma = row.get(f"ma_{w}", yhat)
            row[f"ma_ratio_{w}"] = yhat / (ma + 1e-9)
        row["rsi_14"] = 50
    return pd.DataFrame(preds)

# ============== UI ==============
st.title("📈 Stock Predictor")
st.caption("Free, educational stock forecasting using Random Forest and Alpha Vantage data.")

col1, col2, col3 = st.columns([1.2,1,1])
with col1:
    market = st.selectbox("Market", ["US (NYSE/NASDAQ)", "India (NSE)"])
with col2:
    ticker = st.text_input("Ticker", "AAPL" if market.startswith("US") else "RELIANCE.NS").strip().upper()
with col3:
    horizon = st.slider("Prediction Days", 1, 30, 7)

period = st.radio("Data Range", ["6M", "1Y", "2Y", "5Y"], horizontal=True)
lookback_days = {"6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825}[period]

st.divider()

k1, k2, k3, k4 = st.columns(4)
k1.metric("📅 Range", period)
k2.metric("🔮 Horizon", f"{horizon} day(s)")
k3.metric("API Key", "✅" if get_api_key() else "❌")
k4.metric("Model", "RandomForest")

try:
    df = fetch_alpha_vantage_daily(ticker, full=lookback_days > 100)
    df = df[df.index >= pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)]
    if len(df) < 60:
        st.warning("Not enough data in the selected range.")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price'
    ))
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
    st.subheader("Price Chart")
    st.plotly_chart(fig, use_container_width=True)

    X = make_features(df)
    X_train, X_test, y_train, y_test, features = time_series_train_test_split(X)

    if len(X_train) < 50:
        st.error("Insufficient history to train.")
        st.stop()

    model = build_model()
    model.fit(X_train[features], y_train)

    y_pred = model.predict(X_test[features])
    st.subheader("Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{math.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    m2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
    m3.metric("R²", f"{r2_score(y_test, y_pred):.3f}")

    cmp_fig = go.Figure()
    cmp_fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual'))
    cmp_fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Predicted'))
    cmp_fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
    st.subheader("Prediction vs Actual")
    st.plotly_chart(cmp_fig, use_container_width=True)

    st.subheader(f"{horizon}-Day Forecast")
    forecast_df = forecast_next_days(model, X.iloc[-1][features], horizon)
    st.dataframe(forecast_df)

    csv = df.to_csv().encode()
    st.download_button("Download Data as CSV", csv, "stock_data.csv", "text/csv")

except Exception as e:
    st.error(str(e))
