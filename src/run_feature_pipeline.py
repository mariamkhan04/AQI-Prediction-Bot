# Purpose: End-to-end automation of the Feature Pipeline

import os
import pandas as pd
from datetime import datetime

# --- Import project modules safely ---
try:
    from src.config import SAVE_LOCAL, AIR_QUALITY_URL, WEATHER_FORECAST_URL
    from src.fetch_data import fetch_api_data
    from src.process_data import process_latest_json
    from src.clean_data import clean_data
    from src.process_features import add_features
    from src.upload_to_hopswork import upload_to_hopsworks
except ModuleNotFoundError:
    from config import SAVE_LOCAL, AIR_QUALITY_URL, WEATHER_FORECAST_URL
    from fetch_data import fetch_api_data
    from process_data import process_latest_json
    from clean_data import clean_data
    from process_features import add_features
    from upload_to_hopswork import upload_to_hopsworks


# 1. Pipeline Start 
print("/n🚀 Starting Daily Feature Pipeline for Karachi AQI/n")

try:
    # 2. Step 1: Fetch Latest Raw Data
    print("\n🌤️ Fetching latest Air Quality + Weather data...")
    
    # Fetch raw JSON dictionaries
    aq_json = fetch_api_data(AIR_QUALITY_URL)
    wx_json = fetch_api_data(WEATHER_FORECAST_URL)

    # Convert to DataFrames safely
    aq_df = pd.DataFrame(aq_json["hourly"])
    wx_df = pd.DataFrame(wx_json["hourly"])

    # Add datetime column (for merging)
    if "time" in aq_df.columns and "time" in wx_df.columns:
        aq_df.rename(columns={"time": "datetime"}, inplace=True)
        wx_df.rename(columns={"time": "datetime"}, inplace=True)

    # Merge both datasets on time/datetime
    raw_df = pd.merge(aq_df, wx_df, on="datetime", how="inner")
    print(f"✅ Combined raw data fetched with shape: {raw_df.shape}")

    # 3. Step 2: Process Raw Data
    print("\n⚙️ Processing raw JSON data into structured DataFrame...")
    processed_df = process_latest_json(raw_df)
    print(f"✅ Processed data shape: {processed_df.shape}")

    # 4. Step 3: Clean Data (EDA-1 logic) 
    print("\n🧹 Cleaning processed data...")
    cleaned_df = clean_data(processed_df)
    print(f"✅ Cleaned data shape: {cleaned_df.shape}")

    # 5. Step 4: Feature Engineering (EDA-2 logic) 
    print("\n🧠 Generating engineered features...")
    featured_df = add_features(cleaned_df)
    print(f"✅ Feature engineering complete — shape: {featured_df.shape}")

    # 6. Step 5: Upload to Hopsworks
    print("\n📦 Uploading final dataset to Hopsworks Feature Store...")
    upload_to_hopsworks(featured_df)

    # 7. Verification Step: Read data back from Feature Store
    print("\n🔍 Verifying uploaded data from Feature Store...")
    try:
        import hopsworks

        project = hopsworks.login()
        fs = project.get_feature_store()
        fg = fs.get_feature_group("aqi_features", version=2)

        df_check = fg.read()  # Read full feature group
        df_check["datetime_str"] = pd.to_datetime(df_check["datetime_str"])

        # Sort chronologically to check range
        df_check.sort_values("datetime_str", inplace=True)
        print("\n🧭 Feature Store Data Time Range:")
        print(f"Start → {df_check['datetime_str'].min()}")
        print(f"End   → {df_check['datetime_str'].max()}")

        # Display small samples
        print("\n📊 Head of Feature Store:")
        print(df_check.head(3))
        print("\n📊 Tail of Feature Store:")
        print(df_check.tail(3))
    except Exception as e:
        print("⚠️ Could not verify data from Feature Store:")
        print(str(e))

    print("\n🎉 Feature pipeline executed successfully!")

except Exception as e:
    print("\n❌ Pipeline failed due to error:")
    print(str(e))

finally:
    print("\n🕒 Completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("==============================================")
