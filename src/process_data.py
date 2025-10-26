import os
import json
import glob
import pandas as pd
from config import RAW_PATH, PROCESSED_PATH

def process_latest_json():
    """Convert latest raw JSON (pollutants + weather) to processed CSV"""
    
    # Find latest raw combined JSON file
    files = sorted(glob.glob(f"{RAW_PATH}/raw_combined_*.json"))
    if not files:
        print("❌ No raw JSON files found in", RAW_PATH)
        return
    
    latest_file = files[-1]
    print(f"📂 Using latest file: {os.path.basename(latest_file)}")

    # Load data
    with open(latest_file, "r") as f:
        data = json.load(f)

    # Extract air quality + weather data
    aq = data.get("air_quality", {}).get("hourly", {})
    wx = data.get("weather", {}).get("hourly", {})

    if not aq or not wx:
        print("⚠️ Missing air quality or weather data in JSON.")
        return

    # Convert to DataFrame
    df_aq = pd.DataFrame(aq)
    df_wx = pd.DataFrame(wx)

    # Merge both datasets on 'time'
    df = pd.merge(df_aq, df_wx, on="time", how="inner")
    df.rename(columns={"time": "datetime"}, inplace=True)

    # Save processed raw CSV
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    out_file = os.path.join(
        PROCESSED_PATH,
        f"processed_{df['datetime'].min()[:10]}.csv"
    )

    df.to_csv(out_file, index=False)
    print(f"✅ Processed data saved → {out_file}")
    print(f"📊 Shape: {df.shape}")

if __name__ == "__main__":
    process_latest_json()