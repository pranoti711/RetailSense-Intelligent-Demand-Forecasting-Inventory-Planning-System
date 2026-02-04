import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting preprocessing pipeline...")

PROJECT_ROOT = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\retail forecasting\demand-forecasting-system"

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
EXTERNAL_PATH = os.path.join(PROJECT_ROOT, "data", "external")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

SALES_FILE = os.path.join(RAW_PATH, "train.csv")
HOLIDAY_FILE = os.path.join(EXTERNAL_PATH, "holiday_calendar.csv")
WEATHER_FILE = os.path.join(EXTERNAL_PATH, "weather_data.csv")
TRENDS_FILE = os.path.join(EXTERNAL_PATH, "google_trends.csv")

OUTPUT_FILE = os.path.join(PROCESSED_PATH, "cleaned_sales_data.csv")

print("📁 Sales file:", SALES_FILE)
print("📁 Holiday file:", HOLIDAY_FILE)
print("📁 Weather file:", WEATHER_FILE)
print("📁 Trends file:", TRENDS_FILE)

for file in [SALES_FILE, HOLIDAY_FILE, WEATHER_FILE, TRENDS_FILE]:
    if not os.path.exists(file):
        print(f"⚠️ Warning: File not found: {file}")
    else:
        print(f"✅ Found: {file}")

sales = pd.read_csv(SALES_FILE)
holidays = pd.read_csv(HOLIDAY_FILE) if os.path.exists(HOLIDAY_FILE) else None
weather = pd.read_csv(WEATHER_FILE) if os.path.exists(WEATHER_FILE) else None
trends = pd.read_csv(TRENDS_FILE) if os.path.exists(TRENDS_FILE) else None

print(f"✅ Sales data loaded: {sales.shape}")

def clean_columns(df):
    if df is None:
        return None
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    return df

sales = clean_columns(sales)
holidays = clean_columns(holidays)
weather = clean_columns(weather)
trends = clean_columns(trends)

def parse_date_column(df):
    if df is None:
        return df
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if not date_cols:
        print(f"⚠️ Warning: No date column found in dataframe with columns: {df.columns.tolist()}")
        return df
    date_col = date_cols[0]
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if date_col != 'date':
        df = df.rename(columns={date_col: 'date'})
    return df

sales = parse_date_column(sales)
holidays = parse_date_column(holidays)
weather = parse_date_column(weather)
trends = parse_date_column(trends)
if 'date' in sales.columns:
    sales = sales.dropna(subset=['date'])

if 'date' in sales.columns:
    sales = sales.sort_values("date").reset_index(drop=True)
if holidays is not None and 'date' in holidays.columns:
    holidays = holidays.sort_values("date").reset_index(drop=True)
if weather is not None and 'date' in weather.columns:
    weather = weather.sort_values("date").reset_index(drop=True)
if trends is not None and 'date' in trends.columns:
    trends = trends.sort_values("date").reset_index(drop=True)

df = sales.copy()

if holidays is not None and 'date' in holidays.columns:
    print("🔗 Merging holidays...")
    df = pd.merge(df, holidays, on="date", how="left")
    if "is_holiday" in df.columns:
        df["is_holiday"] = df["is_holiday"].fillna(0)
    if "holiday_name" in df.columns:
        df["holiday_name"] = df["holiday_name"].fillna("None")

if weather is not None and 'date' in weather.columns:
    print("🔗 Merging weather...")
    df = pd.merge(df, weather, on="date", how="left")
    weather_cols = [c for c in weather.columns if c != "date"]
    for col in weather_cols:
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

if trends is not None and 'date' in trends.columns:
    print("🔗 Merging trends...")
    df = pd.merge(df, trends, on="date", how="left")
    trend_cols = [c for c in trends.columns if c != "date"]
    for col in trend_cols:
        df[col] = df[col].fillna(0)

print("✅ Merge complete. Shape:", df.shape)

# ===== HANDLE MISSING TARGET VALUES =====
target_candidates = ["sales", "total_sales", "weekly_sales", "units_sold", "quantity"]
target_col = next((c for c in target_candidates if c in df.columns), None)

if target_col:
    initial_rows = len(df)
    df = df.dropna(subset=[target_col])  
    dropped_rows = initial_rows - len(df)
    print(f"✅ Dropped {dropped_rows} rows with missing {target_col}")
else:
    print("⚠️ WARNING: No known sales target column found!")

df = df.sort_values("date").reset_index(drop=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("🎉 Preprocessing complete!")
print("📦 Cleaned dataset saved to:", OUTPUT_FILE)
print("📊 Date range:", df['date'].min(), "→", df['date'].max())
