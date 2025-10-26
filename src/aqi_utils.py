import math
import numpy as np

# --- Molecular weights (g/mol) for conversions ---
MW = {
    "co": 28.01,       # carbon monoxide
    "no2": 46.0055,    # nitrogen dioxide
    "o3": 48.00,       # ozone
    "so2": 64.066,     # sulfur dioxide
    # PM are mass-based so no MW needed
}

# --- EPA Breakpoints (from AirNow Technical Assistance Document) ---
BP_PM25 = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

BP_PM10 = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

BP_O3_8H = [
    (0, 54, 0, 50),
    (55, 70, 51, 100),
    (71, 85, 101, 150),
    (86, 105, 151, 200),
    (106, 200, 201, 300),
]

# Ozone 1-hour breakpoints (for AQI > 300 case)
BP_O3_1H = [
    (125, 164, 101, 150),
    (165, 204, 151, 200),
    (205, 404, 201, 300),
    (405, 504, 301, 400),
    (505, 604, 401, 500),
]

BP_NO2_1H = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]

BP_SO2_1H = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
    (605, 804, 301, 400),
    (805, 1004, 401, 500),
]

BP_CO_8H = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

# --- Truncation functions (EPA requirement) ---
def truncate(value, pollutant):
    """Truncate pollutant concentration per EPA rules."""
    if value is None:
        return None
    if pollutant == "o3":
        return math.floor(value * 1000) / 1000  # 3 decimals
    elif pollutant == "pm25":
        return math.floor(value * 10) / 10      # 1 decimal
    elif pollutant == "co":
        return math.floor(value * 10) / 10      # 1 decimal
    elif pollutant in ["pm10", "so2", "no2"]:
        return math.floor(value)                # integer
    return value

# --- helper: linear interpolation formula ---
def linear_interpolate(C, C_low, C_high, I_low, I_high):
    return (I_high - I_low) / (C_high - C_low) * (C - C_low) + I_low

def find_bp(C, breakpoints):
    for (C_low, C_high, I_low, I_high) in breakpoints:
        if C_low <= C <= C_high:
            return (C_low, C_high, I_low, I_high)
    return None

# --- conversions ---
def ugm3_to_ppb(ugm3, mw, temp_c=25.0, pressure_hpa=1013.25):
    if ugm3 is None:
        return None
    T_K = temp_c + 273.15
    ppb = ugm3 * (24.45 / mw) * (T_K / 298.15) * (1013.25 / pressure_hpa)
    return ppb

def ugm3_to_ppm_co(ugm3, temp_c=25.0, pressure_hpa=1013.25):
    ppb = ugm3_to_ppb(ugm3, MW["co"], temp_c=temp_c, pressure_hpa=pressure_hpa)
    return ppb / 1000.0 if ppb is not None else None

# --- pollutant AQI functions ---
def aqi_from_conc(conc, breakpoints):
    if conc is None or np.isnan(conc):
        return None
    bp = find_bp(conc, breakpoints)
    if not bp:
        return 500
    C_low, C_high, I_low, I_high = bp
    return linear_interpolate(conc, C_low, C_high, I_low, I_high)

# --- wrapper: compute AQI ---
def compute_aqi_from_row(row, temp_c=25.0, pressure_hpa=1013.25):
    results = {}

    # Truncate concentrations first
    pm25 = truncate(row.get("pm2_5"), "pm25")
    pm10 = truncate(row.get("pm10"), "pm10")
    co_ug = row.get("carbon_monoxide")
    no2_ug = row.get("nitrogen_dioxide")
    o3_ug = row.get("ozone")
    so2_ug = row.get("sulphur_dioxide")

    # AQI for PM
    results["aqi_pm25"] = aqi_from_conc(pm25, BP_PM25)
    results["aqi_pm10"] = aqi_from_conc(pm10, BP_PM10)

    # Convert gases to ppb/ppm and truncate
    no2_ppb = truncate(ugm3_to_ppb(no2_ug, MW["no2"], temp_c, pressure_hpa), "no2") if no2_ug else None
    o3_ppb = truncate(ugm3_to_ppb(o3_ug, MW["o3"], temp_c, pressure_hpa), "o3") if o3_ug else None
    so2_ppb = truncate(ugm3_to_ppb(so2_ug, MW["so2"], temp_c, pressure_hpa), "so2") if so2_ug else None
    co_ppm = truncate(ugm3_to_ppm_co(co_ug, temp_c, pressure_hpa), "co") if co_ug else None

    results["no2_ppb"] = no2_ppb
    results["o3_ppb"] = o3_ppb
    results["so2_ppb"] = so2_ppb
    results["co_ppm"] = co_ppm

    # AQIs for gases
    results["aqi_no2"] = aqi_from_conc(no2_ppb, BP_NO2_1H)
    results["aqi_o3"] = aqi_from_conc(o3_ppb, BP_O3_8H)
    results["aqi_so2"] = aqi_from_conc(so2_ppb, BP_SO2_1H)
    results["aqi_co"] = aqi_from_conc(co_ppm, BP_CO_8H)

    # Special case: O3 > 300 -> use 1-hour values if available
    if results["aqi_o3"] and results["aqi_o3"] > 300:
        results["aqi_o3_1h"] = aqi_from_conc(o3_ppb, BP_O3_1H)
        if results["aqi_o3_1h"]:
            results["aqi_o3"] = max(results["aqi_o3"], results["aqi_o3_1h"])

    # Final AQI = max of all sub-indices
    sub_indices = [results.get(k) for k in ["aqi_pm25","aqi_pm10","aqi_no2","aqi_o3","aqi_so2","aqi_co"]]
    sub_indices_valid = [x for x in sub_indices if x is not None and not (isinstance(x, float) and math.isnan(x))]
    final_aqi = max(sub_indices_valid) if sub_indices_valid else None

    # Round to nearest integer (EPA rule)
    results["aqi"] = round(final_aqi) if final_aqi is not None else None

    return results