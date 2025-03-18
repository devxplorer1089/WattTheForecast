import pandas as pd
import numpy as np
import os

# =========================
# Define Paths
# =========================
DATA_DIR = "datasets/optimized"
OUTPUT_DIR = "datasets/finalized"
PLOTS_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================
# Load Preprocessed Data
# =========================
files = {
    "price": "optimized_refined_Day-ahead_prices_202301010000_202503050000_Hour.csv",
    "actual_consumption": "optimized_refined_Actual_consumption_202301010000_202503050000_Quarterhour.csv",
    "forecast_consumption": "optimized_refined_Forecasted_consumption_202301010000_202503050000_Quarterhour.csv",
    "actual_generation": "optimized_refined_Actual_generation_202301010000_202503050000_Quarterhour.csv",
    "forecast_generation": "optimized_refined_Forecasted_generation_Day-Ahead_202301010000_202503050000_Hour_Quarterhour.csv",
    "cross_border_flows": "optimized_refined_Cross-border_physical_flows_202301010000_202503050000_Quarterhour.csv",
    "scheduled_exchanges": "optimized_refined_Scheduled_commercial_exchanges_202301010000_202503050000_Quarterhour.csv",
}

# Load datasets
datasets = {key: pd.read_csv(os.path.join(DATA_DIR, file), delimiter=",", low_memory=False) for key, file in files.items()}

# =========================
# Clean Column Names
# =========================
for df in datasets.values():
    df.columns = df.columns.str.strip().str.replace(r"[^\x00-\x7F]+", "", regex=True)  # Remove spaces & non-ASCII chars

# =========================
# Convert "-" to NaN and Ensure Numeric Columns
# =========================
for df in datasets.values():
    df.replace("-", np.nan, inplace=True)
    df.infer_objects(copy=False)

    for col in df.columns:
        if col != "Start date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# Compute Average Price
# =========================
df_price = datasets["price"]
price_columns = [col for col in df_price.columns if "/MWh" in col]

if not price_columns:
    raise KeyError("No valid price columns found in dataset.")

df_price["Average_Price_€/MWh"] = df_price[price_columns].mean(axis=1)

# =========================
# Drop Duplicate Columns Before Merge
# =========================
for df in datasets.values():
    df.drop(columns=["End date"], errors="ignore", inplace=True)

# =========================
# Convert Time Columns
# =========================
for df in datasets.values():
    df["Start date"] = pd.to_datetime(df["Start date"], errors="coerce")

# =========================
# Merge Datasets
# =========================
df = df_price.copy()

for key, df_other in datasets.items():
    if key != "price":
        df = df.merge(df_other, on="Start date", how="inner", suffixes=("", f"_{key}"))

# Ensure unique column names
df = df.loc[:, ~df.columns.duplicated()]

# =========================
# Compute Total Forecast Generation
# =========================
forecast_gen_cols = [col for col in df.columns if "forecast_generation" in col]

if not forecast_gen_cols:
    raise KeyError("No forecast generation columns found.")

df["Total_Forecast_Generation"] = df[forecast_gen_cols].sum(axis=1)

# Compute Generation Imbalance
if "Total [MWh] Original resolutions" in df.columns:
    df["Generation_Imbalance"] = df["Total [MWh] Original resolutions"] - df["Total_Forecast_Generation"]
else:
    raise KeyError("'Total [MWh] Original resolutions' column is missing.")

# =========================
# Handle Missing Values
# =========================
df.fillna(df.median(), inplace=True)

# Recompute Average Price
df["Average_Price_€/MWh"] = df[price_columns].mean(axis=1)

# =========================
# Feature Engineering
# =========================
df["Rolling_Mean_24h"] = df["Average_Price_€/MWh"].rolling(window=24, min_periods=1).mean()
df["Rolling_Mean_7d"] = df["Average_Price_€/MWh"].rolling(window=24 * 7, min_periods=1).mean()
df["Price_Diff"] = df["Average_Price_€/MWh"].diff()
df["Lag_1h"] = df["Average_Price_€/MWh"].shift(1)
df["Lag_24h"] = df["Average_Price_€/MWh"].shift(24)
df["Volatility_24h"] = df["Average_Price_€/MWh"].rolling(window=24, min_periods=1).std()
df["Price_Change_1h"] = df["Average_Price_€/MWh"].pct_change() * 100
df["Price_Change_24h"] = df["Average_Price_€/MWh"].pct_change(24) * 100

# Compute Consumption Imbalance
if "Total (grid load) [MWh] Original resolutions" in df.columns and "Total (grid load) [MWh] Original resolutions_forecast_consumption" in df.columns:
    df["Consumption_Imbalance"] = df["Total (grid load) [MWh] Original resolutions"] - df["Total (grid load) [MWh] Original resolutions_forecast_consumption"]

# =========================
# Aggregate Data
# =========================
df.set_index("Start date", inplace=True)
df_hourly = df.resample("H").mean()
df_daily = df.resample("D").mean()
df_weekly = df.resample("W").mean()

# =========================
# Save Processed Data
# =========================
df_hourly.to_csv(os.path.join(OUTPUT_DIR, "finalized_hourly_data.csv"), sep=",")
df_daily.to_csv(os.path.join(OUTPUT_DIR, "finalized_daily_data.csv"), sep=",")
df_weekly.to_csv(os.path.join(OUTPUT_DIR, "finalized_weekly_data.csv"), sep=",")

print("Data processing completed successfully.")
