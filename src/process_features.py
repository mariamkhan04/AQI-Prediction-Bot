# src/process_features.py
import pandas as pd
import numpy as np
import os

# Safe import (works both in root or src/)
try:
    from src.aqi_utils import compute_aqi_from_row
except Exception:
    from aqi_utils import compute_aqi_from_row


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute AQI, time-based, and derived features for ML training.
    Includes both Phase-1 (basic) and Phase-2 (advanced) engineered features.
    """

    df = df.copy()

    # --- 1️⃣ Normalize datetime column ---
    if "time" in df.columns and "datetime" not in df.columns:
        df.rename(columns={"time": "datetime"}, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- 2️⃣ Compute AQI (using compute_aqi_from_row) ---
    print("⚙️ Computing AQI and sub-indices...")
    aqi_results = df.apply(lambda row: compute_aqi_from_row(row), axis=1)

    # Expand dict results into DataFrame
    if isinstance(aqi_results.iloc[0], dict):
        aqi_expanded = pd.DataFrame(list(aqi_results))
        aqi_expanded = aqi_expanded.apply(pd.to_numeric, errors="coerce")
        df = pd.concat([df.reset_index(drop=True), aqi_expanded.reset_index(drop=True)], axis=1)
    else:
        df["aqi"] = pd.to_numeric(aqi_results, errors="coerce")

    # Ensure 'aqi' column exists
    if "aqi" not in df.columns:
        sub_cols = [c for c in df.columns if c.startswith("aqi_")]
        df["aqi"] = df[sub_cols].max(axis=1) if sub_cols else np.nan

    # Drop rows where AQI couldn't be computed
    df.dropna(subset=["aqi"], inplace=True)

    # --- 3️⃣ Time-based features ---
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    # Cyclic (sin/cos) encoding for hours (to capture 24h periodicity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # --- 4️⃣ Derived features ---
    df["aqi_change_rate"] = df["aqi"].diff()
    df["aqi_roll_mean_3h"] = df["aqi"].rolling(window=3, min_periods=1).mean()
    df["aqi_roll_mean_6h"] = df["aqi"].rolling(window=6, min_periods=1).mean()
    df["aqi_rolling_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()

    # --- 5️⃣ Lag features (previous hours AQI) ---
    for lag in [1, 3, 6]:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)

    # --- 6️⃣ Pollutant ratio features ---
    df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)

    # --- 7️⃣ Meteorological combination features ---
    df["temp_humidity_ratio"] = df["temperature_2m"] / (df["relative_humidity_2m"] + 1e-6)

    # Wind vector components — pollutant dispersion
    df["wind_effect"] = df["wind_speed_10m"] * np.cos(np.deg2rad(df["wind_direction_10m"]))

    # --- 8️⃣ High pollution flag (for alert classification tasks) ---
    df["high_pollution_flag"] = np.where(df["aqi"] > 150, 1, 0)

    # --- 9️⃣ Handle any NaNs from lags/ratios (optional fill) ---
    df.ffill(inplace=True)
    df.bfill(inplace=True)


    print("✅ Feature engineering complete!")
    print(f"Final shape: {df.shape}")

    return df

# --- Run standalone test safely ---
if __name__ == "__main__":
    import os

    # 🧭 Step 1: Determine absolute paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "final", "clean_merged_karachi.csv")
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "final", "final_features.csv")

    # 🧪 Step 2: Check if input exists
    if os.path.exists(INPUT_PATH):
        print(f"📂 Found input file: {INPUT_PATH}")
        df = pd.read_csv(INPUT_PATH)

        print("⚙️ Running feature engineering...")
        df = add_features(df)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Final engineered dataset saved successfully → {OUTPUT_PATH}")
        print(f"Final shape: {df.shape}")
        print(df.columns.tolist())
    else:
        print(f"❌ Input file not found → {INPUT_PATH}")
        print("💡 Please run '01_eda_preprocessing.ipynb' first to generate clean_merged_karachi.csv.")