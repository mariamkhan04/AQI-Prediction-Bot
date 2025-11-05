# Purpose: Load latest data from Hopsworks, predict AQI for next 3 days and evaluate model performance on test data.
import hopsworks
import pandas as pd
import numpy as np
import os
from joblib import load
from datetime import timedelta

# 1. Connect to Hopsworks Feature Store
print("🔗 Connecting to Hopsworks Feature Store...")
import dotenv
dotenv.load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()
print("✅ Data fetched from Hopsworks successfully!")

print(f"Initial shape: {df.shape}")

# 2. Prepare data
if "datetime_str" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime_str"])
    df.drop(columns=["datetime_str"], inplace=True)

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# 3. Drop leakage features (must match training) 
leakage_features = ["aqi_rolling_24h", "aqi_lag_1h", "high_pollution_flag"]
for col in leakage_features:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"⚠️ Dropped leakage feature: {col}")

# 4. Split features and labels for evaluation
X = df.drop(columns=["aqi", "datetime"], errors="ignore")
y = df["aqi"]

# 5. Load trained model 
model_path = os.path.join(os.path.dirname(__file__), "../models/best_model_random_forest.pkl")
model = load(model_path)
print(f"Loaded trained model from {model_path}")

# 6. Evaluate model on existing data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

preds = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, preds))
mae = mean_absolute_error(y, preds)
r2 = r2_score(y, preds)

print("\n📊 Model Evaluation on Full Data ---")
print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# 7. Predict next 3 days AQI (72 hours ahead) 
print("\n📆 Generating next 3 days hourly AQI predictions...")

last_date = df["datetime"].max()
future_dates = [last_date + timedelta(hours=i) for i in range(1, 73)]

base_features = X.iloc[-1].copy()

# Slight variation for realism
vary_cols = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "ozone", "sulphur_dioxide", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m"
]

future_data = pd.DataFrame([base_features for _ in range(72)])
for col in vary_cols:
    if col in future_data.columns:
        noise = np.random.normal(0, 0.05, size=72)  # ±5% variation
        future_data[col] = future_data[col] * (1 + noise)

future_data["datetime"] = future_dates
future_preds = model.predict(future_data.drop(columns=["datetime"], errors="ignore"))

future_results = pd.DataFrame({
    "datetime": future_dates,
    "predicted_AQI": future_preds
})

# 8. Save hourly predictions
output_path = os.path.join(os.path.dirname(__file__), "../data/predictions/next_3_days_predictions.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
future_results.to_csv(output_path, index=False)
print(f"💾 Saved hourly predictions to {output_path}")

# 9. Print daily averages 
future_results["date"] = future_results["datetime"].dt.date
daily_avg = future_results.groupby("date")["predicted_AQI"].mean().reset_index()

print("\n📅 Average Predicted AQI for Next 3 Days ---")
for _, row in daily_avg.iterrows():
    print(f"📆 {row['date']} → Average AQI: {row['predicted_AQI']:.2f}")

# 10. Trend Validation
try:
    # Load last 24h from feature group for comparison
    df_recent = df.sort_values("datetime").tail(24)
    avg_actual_aqi = df_recent["aqi"].mean()
    avg_predicted_aqi = future_results.head(24)["predicted_AQI"].mean()

    print(f"\n📊 Avg Actual AQI (last 24h): {avg_actual_aqi:.2f}")
    print(f"📊 Avg Predicted AQI (next 24h): {avg_predicted_aqi:.2f}")

    if avg_predicted_aqi > avg_actual_aqi + 5:
        print("🚨 Air quality expected to worsen slightly in the next 24h.")
    elif avg_predicted_aqi < avg_actual_aqi - 5:
        print("Air quality expected to improve slightly in the next 24h.")
    else:
        print("Air quality expected to remain stable in the next 24h.")
except Exception as e:
    print("Trend comparison skipped due to:", e)

print("\nPrediction and Evaluation Complete!")