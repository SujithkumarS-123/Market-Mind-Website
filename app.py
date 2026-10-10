from __future__ import annotations

import warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "Market_Mind_25_Stock_CNN_LSTM_V2.keras"
PREPROCESS_PATH = ARTIFACT_DIR / "Market_Mind_Preprocessing.joblib"
TARGET_SCALER_PATH = ARTIFACT_DIR / "Market_Mind_V2_Target_Scaler.joblib"
MAPPING_PATH = ARTIFACT_DIR / "Market_Mind_Stock_Mapping.joblib"
EXPECTED_FEATURES = [
    "Daily_Return", "Open_Gap", "High_Low_Range", "Close_Position",
    "Volume_Change", "Close_vs_SMA5", "Close_vs_SMA20", "Volatility_5",
]
LOOKBACK_DEFAULT = 60

st.set_page_config(page_title="Market Mind | AI Stock Analytics", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 10% 0%, #10233c 0, #07111f 36%, #050b14 100%); color: #eaf2ff; }
.block-container { padding-top: 1.6rem; max-width: 1500px; }
.mm-title { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; letter-spacing: -.04em; color: #f5f8ff; }
.mm-subtitle { color: #8ea4c3; margin-top: -.4rem; }
.mm-pill { display:inline-block; padding: 5px 10px; border:1px solid #214c43; background:#0b2a25; color:#54e0a2; border-radius:999px; font-size:.78rem; }
[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(19,37,60,.96), rgba(11,23,39,.96)); border: 1px solid #1f3551; border-radius: 16px; padding: 18px 18px 14px; }
[data-testid="stMetricLabel"] { color: #94a9c5 !important; }
[data-testid="stMetricValue"] { color: #f5f8ff !important; font-family:'Space Grotesk',sans-serif; }
[data-testid="stMetricDelta"] svg { display: none; }
section[data-testid="stSidebar"] { background: #091422; border-right:1px solid #1c2d43; }
section[data-testid="stSidebar"] * { color: #eaf2ff; }
div.stButton > button { background: linear-gradient(90deg,#19b981,#0f9f78); color:white; border:0; border-radius:10px; font-weight:700; min-height:2.8rem; }
div.stButton > button:hover { background: #22c997; color:white; border:0; }
hr { border-color: #1b2c42; }
.small-note { color:#8298b6; font-size:.83rem; }
</style>
""", unsafe_allow_html=True)


def missing_artifacts() -> list[str]:
    return [p.name for p in [MODEL_PATH, PREPROCESS_PATH, TARGET_SCALER_PATH, MAPPING_PATH] if not p.exists()]


@st.cache_resource(show_spinner="Loading saved Market Mind model...")
def load_assets():
    absent = missing_artifacts()
    if absent:
        raise FileNotFoundError("Missing artifact files in the artifacts folder: " + ", ".join(absent))
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    preprocessing = joblib.load(PREPROCESS_PATH)
    mapping = joblib.load(MAPPING_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    feature_scaler = preprocessing["scaler"]
    features = list(preprocessing["features"])
    stock_to_id = mapping.get("stock_to_id", preprocessing.get("stock_to_id", {}))
    lookback = int(mapping.get("lookback", LOOKBACK_DEFAULT))
    if features != EXPECTED_FEATURES:
        raise ValueError(f"Feature order in saved preprocessing does not match the dashboard. Saved: {features}")
    if not stock_to_id:
        raise ValueError("No stock_to_id mapping found in saved mapping/preprocessing artifacts.")
    return model, feature_scaler, target_scaler, stock_to_id, features, lookback


def download_stock(ticker: str, lookback: int) -> pd.DataFrame:
    raw = yf.download(ticker, period="max", interval="1d", auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError(f"No market data was returned for {ticker}. Check your internet connection or Yahoo Finance availability.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Market data is missing columns: {missing}")
    raw = raw[required].copy()
    raw.index = pd.to_datetime(raw.index)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index().dropna(subset=required)
    today_india = pd.Timestamp(datetime.now(ZoneInfo("Asia/Kolkata")).date())
    raw = raw.loc[raw.index.normalize() < today_india]
    if len(raw) < lookback + 30:
        raise ValueError("There is not enough completed historical market data for this prediction.")
    return raw


def prepare_features(raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = raw.copy()
    prev_close = df["Close"].shift(1)
    df["Daily_Return"] = df["Close"].pct_change()
    df["Open_Gap"] = df["Open"] / prev_close - 1
    df["High_Low_Range"] = (df["High"] - df["Low"]) / prev_close
    day_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["Close_Position"] = (df["Close"] - df["Low"]) / day_range
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Close_vs_SMA5"] = df["Close"] / df["Close"].rolling(5).mean() - 1
    df["Close_vs_SMA20"] = df["Close"] / df["Close"].rolling(20).mean() - 1
    df["Volatility_5"] = df["Daily_Return"].rolling(5).std()
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    return df


def predict_returns(model, target_scaler, X: np.ndarray, stock_id: int) -> np.ndarray:
    ids = np.full((len(X), 1), stock_id, dtype=np.int32)
    output = model.predict([X, ids], verbose=0).reshape(-1, 1)
    return target_scaler.inverse_transform(output).ravel()


def build_results(stock: str, ticker: str, assets):
    model, feature_scaler, target_scaler, stock_to_id, features, lookback = assets
    if stock not in stock_to_id:
        raise ValueError(f"{stock} is not included in the saved model's stock mapping.")
    raw = download_stock(ticker, lookback)
    df = prepare_features(raw, features)
    valid = df.dropna(subset=features).copy()
    if len(valid) <= lookback + 20:
        raise ValueError("Not enough valid feature rows after preprocessing.")
    stock_id = int(stock_to_id[stock])

    latest_window = valid[features].iloc[-lookback:]
    X_live = feature_scaler.transform(latest_window).astype(np.float32).reshape(1, lookback, len(features))
    latest_date = valid.index[-1]
    latest = valid.loc[latest_date]
    latest_close = float(latest["Close"])
    latest_open = float(latest["Open"])
    predicted_return = float(predict_returns(model, target_scaler, X_live, stock_id)[0])
    predicted_close = latest_close * (1 + predicted_return)

    scaled_all = feature_scaler.transform(valid[features]).astype(np.float32)
    closes = valid["Close"].to_numpy(dtype=float)
    dates = valid.index
    sequences, origin_close, actual_next_close, prediction_dates = [], [], [], []
    for i in range(lookback - 1, len(valid) - 1):
        window = scaled_all[i - lookback + 1:i + 1]
        if not np.isfinite(window).all():
            continue
        sequences.append(window)
        origin_close.append(closes[i])
        actual_next_close.append(closes[i + 1])
        prediction_dates.append(dates[i + 1])
    X_all = np.asarray(sequences, dtype=np.float32)
    origin_close = np.asarray(origin_close, dtype=float)
    actual_next_close = np.asarray(actual_next_close, dtype=float)
    prediction_dates = pd.to_datetime(prediction_dates)
    split = int(len(X_all) * 0.80)
    X_test, origins_test, actual_test = X_all[split:], origin_close[split:], actual_next_close[split:]
    test_dates = prediction_dates[split:]
    if len(X_test) < 10:
        raise ValueError("Not enough historical test sequences to calculate evaluation metrics.")
    predicted_test_returns = predict_returns(model, target_scaler, X_test, stock_id)
    predicted_test = origins_test * (1 + predicted_test_returns)
    baseline_test = origins_test
    residuals = actual_test - predicted_test
    metrics = {
        "Test predictions": int(len(actual_test)),
        "MAE": float(mean_absolute_error(actual_test, predicted_test)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_test, predicted_test))),
        "MAPE": float(np.mean(np.abs((actual_test - predicted_test) / actual_test)) * 100),
        "R²": float(r2_score(actual_test, predicted_test)),
        "Directional accuracy": float(np.mean(np.sign(actual_test - origins_test) == np.sign(predicted_test - origins_test)) * 100),
        "Baseline MAE": float(mean_absolute_error(actual_test, baseline_test)),
        "Baseline RMSE": float(np.sqrt(mean_squared_error(actual_test, baseline_test))),
    }
    history = pd.DataFrame({
        "Date": test_dates,
        "Origin_Close": origins_test,
        "Actual_Close": actual_test,
        "Predicted_Close": predicted_test,
        "Baseline_Close": baseline_test,
        "Residual": residuals,
        "Absolute_Error": np.abs(residuals),
    })
    return {
        "raw": raw, "valid": valid, "stock": stock, "ticker": ticker,
        "latest_date": latest_date, "latest_open": latest_open, "latest_close": latest_close,
        "predicted_return": predicted_return, "predicted_close": predicted_close,
        "predicted_change": predicted_close - latest_close,
        "metrics": metrics, "history": history,
    }


def money(v):
    return f"₹{v:,.2f}"

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 📈 Market Mind")
    st.caption("AI-assisted stock analytics")
    st.markdown("---")
    st.markdown("### Prediction settings")
    stocks = sorted([s for s in []])
    try:
        _, _, _, saved_mapping, _, _ = load_assets()
        stocks = sorted(saved_mapping.keys())
        default_stock = stocks.index("BRITANNIA") if "BRITANNIA" in stocks else 0
    except Exception:
        stocks = ["ADANIENT", "ASIANPAINT", "AXISBANK", "BAJAJFINSV", "BRITANNIA", "CIPLA", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HEROMOTOCO", "HINDALCO", "ICICIBANK", "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "MARUTI", "NESTLEIND", "SUNPHARMA", "TCS", "TITAN", "WIPRO"]
        default_stock = stocks.index("BRITANNIA")
    selected_stock = st.selectbox("Choose NSE stock", stocks, index=default_stock)
    st.caption(f"Yahoo Finance symbol: `{selected_stock}.NS`")
    predict_button = st.button("↗  Predict & refresh dashboard", use_container_width=True)
    st.markdown("---")
    st.markdown("**Data source**")
    st.caption("Yahoo Finance daily historical data")
    st.markdown("**Model**")
    st.caption("Saved Market Mind V2 CNN–LSTM")

# -------------------- HEADER --------------------
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown('<div class="mm-title">Market Mind <span style="color:#48e0a4">↗</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="mm-subtitle">AI stock analytics · 25 NSE stocks · Daily market view</div>', unsafe_allow_html=True)
with col_status:
    st.markdown('<div style="text-align:right;padding-top:12px"><span class="mm-pill">● LOCAL DASHBOARD</span></div>', unsafe_allow_html=True)
st.markdown("")

st.info("Choose a stock on the left and click **Predict & refresh dashboard**. Market data is fetched when you request a prediction; the saved model is loaded locally.")

if "mm_result" not in st.session_state:
    st.session_state.mm_result = None
if predict_button:
    try:
        assets = load_assets()
        with st.spinner(f"Fetching completed market sessions and running the saved model for {selected_stock}…"):
            st.session_state.mm_result = build_results(selected_stock, f"{selected_stock}.NS", assets)
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        st.session_state.mm_result = None

result = st.session_state.mm_result
if result is None:
    st.markdown("### Your market workspace is ready")
    st.markdown("Select a stock and run a prediction to load its market history, forecast, charts and historical evaluation.")
    a, b, c = st.columns(3)
    a.metric("Supported stocks", "25")
    b.metric("Model", "CNN–LSTM V2")
    c.metric("Forecast horizon", "Next trading session")
    st.caption("Forecasts are experimental model estimates, not guaranteed market outcomes or investment advice.")
    st.stop()

raw = result["raw"]
history = result["history"]
metrics = result["metrics"]
change = result["predicted_change"]
change_pct = result["predicted_return"] * 100

st.markdown(f"### {result['stock']}  <span style='color:#8ea4c3;font-size:.85rem'>NSE · {result['latest_date']:%d %b %Y}</span>", unsafe_allow_html=True)
price_cols = st.columns(4)
price_cols[0].metric("Latest close", money(result["latest_close"]))
price_cols[1].metric("Predicted next close", money(result["predicted_close"]), f"{change_pct:+.2f}%")
price_cols[2].metric("Estimated price change", money(change), f"{change:+.2f}")
price_cols[3].metric("Model direction", "UP ↑" if change_pct > 0 else ("DOWN ↓" if change_pct < 0 else "FLAT →"), f"{change_pct:+.2f}%")

st.markdown("---")
left, right = st.columns([2.15, 1])
with left:
    st.markdown("### Price action")
    view = raw.tail(180)
    fig = go.Figure(data=[go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
        increasing_line_color="#22c55e", decreasing_line_color="#fb7185", name="OHLC"
    )])
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,18,31,.7)",
                      height=430, margin=dict(l=8,r=8,t=18,b=8), xaxis_rangeslider_visible=False,
                      yaxis_title="Price (INR)", font=dict(color="#c8d7ea"))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown("### Next-session estimate")
    direction_color = "#22c55e" if change_pct >= 0 else "#fb7185"
    st.markdown(f"<div style='background:linear-gradient(145deg,#132a3b,#0d1b2b);border:1px solid #233d54;border-radius:18px;padding:24px 20px'><div style='color:#92a8c3;font-size:.9rem'>Predicted closing price</div><div style='font-size:2rem;font-weight:700;color:#f3f7ff;margin:8px 0'>{money(result['predicted_close'])}</div><div style='font-size:1.15rem;font-weight:700;color:{direction_color}'>{change_pct:+.2f}% · {money(change)} </div><div style='color:#839ab5;font-size:.78rem;margin-top:16px'>Forecast based on the latest completed daily data</div></div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("**Latest session**")
    st.write("Open", money(result["latest_open"]))
    st.write("Close", money(result["latest_close"]))
    st.write("Session date", result["latest_date"].strftime("%d %b %Y"))
    st.download_button("Download historical evaluation CSV", history.to_csv(index=False).encode("utf-8"), file_name=f"market_mind_{result['stock']}_evaluation.csv", mime="text/csv", use_container_width=True)

st.markdown("---")
st.markdown("### Market indicators")
ind1, ind2 = st.columns(2)
with ind1:
    volume_fig = go.Figure(go.Bar(x=raw.tail(60).index, y=raw.tail(60)["Volume"], marker_color="#38bdf8"))
    volume_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,18,31,.7)", height=280, margin=dict(l=8,r=8,t=10,b=8), yaxis_title="Volume")
    st.plotly_chart(volume_fig, use_container_width=True)
with ind2:
    closes = raw["Close"].tail(120)
    sma_fig = go.Figure()
    sma_fig.add_trace(go.Scatter(x=closes.index, y=closes, name="Close", line=dict(color="#38bdf8", width=2)))
    sma_fig.add_trace(go.Scatter(x=closes.index, y=closes.rolling(20).mean(), name="20-session average", line=dict(color="#fbbf24", width=2)))
    sma_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,18,31,.7)", height=280, margin=dict(l=8,r=8,t=10,b=8), yaxis_title="Price (INR)")
    st.plotly_chart(sma_fig, use_container_width=True)

st.markdown("---")
st.markdown("### Historical model evaluation")
st.caption("Chronological 80/20 split using the current downloaded history. These metrics are recalculated for the selected stock; they are not a guarantee of future performance.")
mc = st.columns(4)
mc[0].metric("MAE", money(metrics["MAE"]))
mc[1].metric("RMSE", money(metrics["RMSE"]))
mc[2].metric("MAPE", f"{metrics['MAPE']:.3f}%")
mc[3].metric("Directional accuracy", f"{metrics['Directional accuracy']:.1f}%")
metric_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
st.dataframe(metric_df, hide_index=True, use_container_width=True)

hist_fig = go.Figure()
hist_fig.add_trace(go.Scatter(x=history["Date"], y=history["Actual_Close"], name="Actual close", line=dict(color="#38bdf8", width=2)))
hist_fig.add_trace(go.Scatter(x=history["Date"], y=history["Predicted_Close"], name="Predicted close", line=dict(color="#fbbf24", width=1.6)))
hist_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,18,31,.7)", height=350, margin=dict(l=8,r=8,t=12,b=8), yaxis_title="Price (INR)")
st.plotly_chart(hist_fig, use_container_width=True)

with st.expander("Recent historical rows"):
    st.dataframe(raw.tail(20).sort_index(ascending=False), use_container_width=True)

st.caption("Market Mind is an educational forecasting project. Predictions are experimental estimates, can be wrong, and are not investment advice. Historical evaluation depends on reconstructed feature engineering and the selected data source.")
