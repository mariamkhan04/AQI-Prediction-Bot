from datetime import datetime, timedelta

LAT = 24.900002   # Karachi latitude
LON = 67.0  # Karachi longitude

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
    f"&forecast_days=1"
)

WEATHER_FORECAST_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    f"&forecast_days=1"
)

# base urls for historical data
start_date = "2024-01-01"
end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
aq_historic_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
    )
weather_historic_url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    )

# Data path
RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed"
HIST_PATH = "data/historical"

import os
SAVE_LOCAL = os.getenv("SAVE_LOCAL", "false").lower() in ("1", "true", "yes")