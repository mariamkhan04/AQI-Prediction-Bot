import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import os
from joblib import load
from datetime import timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# PAGE CONFIG 
st.set_page_config(
    page_title="Karachi AQI Prediction Bot",
    page_icon="🌤",
    layout="wide"
)

# INLINE STYLING 
st.markdown("""
    <style>
        body {
            background-color: #f6fafc;
            color: #222;
            font-family: "Inter", sans-serif;
        }
        .main-title {
            text-align: center;
            color: #1565C0;
            font-weight: 800;
            font-size: 32px;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            color: #388E3C;
            font-size: 16px;
            margin-bottom: 25px;
        }
        .metric-card {
            padding: 15px;
            background-color: #E3F2FD;
            border-left: 5px solid #1565C0;
            border-radius: 10px;
            box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
        }
        .chart-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
        }
        .footer {
            text-align: center;
            color: gray;
            font-size: 13px;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# TITLE SECTION 
st.markdown("<h1 class='main-title'>🌆 Karachi AQI Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time and 3-day Air Quality predictions powered by ML & Hopsworks Feature Store.</p>", unsafe_allow_html=True)

# CONNECT TO HOPSWORKS
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

try:
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    st.success("✅ Connected to Hopsworks and fetched latest data.")
except Exception as e:
    st.error("⚠ Could not fetch data from Hopsworks. Using local fallback.")
    df = pd.read_csv("../data/final/final_selected_features.csv")

# DATA PREPARATION
if "datetime_str" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime_str"])
    df.drop(columns=["datetime_str"], inplace=True)

df = df.sort_values("datetime").reset_index(drop=True)

# Drop leakage features
leakage_features = ["aqi_rolling_24h", "aqi_lag_1h", "high_pollution_flag"]
df.drop(columns=[col for col in leakage_features if col in df.columns], inplace=True, errors="ignore")

# Define features
X = df.drop(columns=["aqi", "datetime"], errors="ignore")

# LOAD TRAINED MODEL
try:
    model_path = os.path.join(os.path.dirname(__file__), "../models/best_model_random_forest.pkl")
    model = load(model_path)
    st.success("✅ Loaded latest trained Random Forest model.")
except Exception as e:
    st.error(f"⚠ Could not load model: {e}")
    st.stop()

# CURRENT AQI
today_data = X.iloc[-1:]
today_aqi = model.predict(today_data)[0]

st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
st.subheader("🌤 Current AQI Prediction")
st.metric("Predicted AQI (Current Hour)", f"{today_aqi:.2f}")
# --- AQI CATEGORY INDICATOR ---
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "🟢 Good", "#66BB6A"
    elif aqi_value <= 100:
        return "🟡 Moderate", "#FDD835"
    elif aqi_value <= 150:
        return "🟠 Unhealthy for Sensitive Groups", "#FB8C00"
    elif aqi_value <= 200:
        return "🔴 Unhealthy", "#E53935"
    elif aqi_value <= 300:
        return "🟣 Very Unhealthy", "#8E24AA"
    else:
        return "🟤 Hazardous", "#6D4C41"

category, color = get_aqi_category(today_aqi)

st.markdown(
    f"""
    <div style='background-color:{color}22; border-left: 5px solid {color}; 
         padding: 10px 15px; border-radius:8px; margin-top:10px'>
        <b>Category:</b> {category}
    </div>
    """,
    unsafe_allow_html=True
)

# FUTURE PREDICTIONS
st.subheader("📅 AQI Forecast for Next 3 Days")

last_date = df["datetime"].max()
future_dates = [last_date + timedelta(hours=i) for i in range(1, 73)]
base_features = X.iloc[-1].copy()

# ±5% random variation
vary_cols = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "ozone", "sulphur_dioxide", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m"
]

future_data = pd.DataFrame([base_features for _ in range(72)])
for col in vary_cols:
    if col in future_data.columns:
        noise = np.random.normal(0, 0.05, size=72)
        future_data[col] = future_data[col] * (1 + noise)

future_data["datetime"] = future_dates
future_preds = model.predict(future_data.drop(columns=["datetime"], errors="ignore"))

future_results = pd.DataFrame({
    "datetime": future_dates,
    "predicted_AQI": future_preds
})
future_results["date"] = future_results["datetime"].dt.date

# VISUALS 
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.write("📈 **Hourly AQI Trend (Next 3 Days)**")
    st.line_chart(future_results.set_index("datetime")["predicted_AQI"], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.write("📊 **Daily Average AQI Forecast**")
    daily_avg = future_results.groupby("date")["predicted_AQI"].mean().reset_index()
    st.bar_chart(daily_avg.set_index("date"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

#FORECAST TABLE 
st.subheader("📊 3-Day AQI Forecast Summary")
st.dataframe(daily_avg, hide_index=True, use_container_width=True)

# INTERPRETATION 
st.markdown("---")
avg_aqi_today = df.tail(24)["aqi"].mean() if "aqi" in df.columns else None
avg_pred_aqi_next = future_results.head(24)["predicted_AQI"].mean()

if avg_aqi_today and avg_pred_aqi_next:
    if avg_pred_aqi_next > avg_aqi_today + 5:
        st.warning("🚨 Air quality expected to worsen slightly in the next 24 hours.")
    elif avg_pred_aqi_next < avg_aqi_today - 5:
        st.info("🌿 Air quality expected to improve slightly in the next 24 hours.")
    else:
        st.success("✅ Air quality expected to remain stable in the next 24 hours.")

# FOOTER 
st.markdown("<p class='footer'>Developed by Mariam Khan | Powered by Hopsworks ✨</p>", unsafe_allow_html=True)