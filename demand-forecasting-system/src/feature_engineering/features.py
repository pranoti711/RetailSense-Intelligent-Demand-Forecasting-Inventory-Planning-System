
import pandas as pd
import os

PROJECT_ROOT = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\retail forecasting\demand-forecasting-system"

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURE_PATH = os.path.join(PROJECT_ROOT, "data", "features")

CLEANED_FILE = os.path.join(PROCESSED_PATH, "cleaned_sales_data.csv")

FEATURE_FILE = os.path.join(FEATURE_PATH, "sales_features.csv")

os.makedirs(FEATURE_PATH, exist_ok=True)

print("Starting Feature Engineering...")

df = pd.read_csv(CLEANED_FILE)
df['date'] = pd.to_datetime(df['date'])

df = df.sort_values("date").reset_index(drop=True)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday  
df['is_weekend'] = df['weekday'] >= 5

TARGET_COL = None
for c in ["weekly_sales", "sales", "total_sales", "units_sold", "quantity"]:
    if c in df.columns:
        TARGET_COL = c
        break

if TARGET_COL is None:
    raise ValueError("No known target column found in dataset!")

df['lag_1'] = df[TARGET_COL].shift(1)
df['lag_2'] = df[TARGET_COL].shift(2)
df['lag_3'] = df[TARGET_COL].shift(3)

df['rolling_4w_mean'] = df[TARGET_COL].rolling(window=4, min_periods=1).mean()
df['rolling_4w_std'] = df[TARGET_COL].rolling(window=4, min_periods=1).std().fillna(0)

extra_cols = [c for c in df.columns if c not in ['date', TARGET_COL, 'year','month','day','weekday','is_weekend','lag_1','lag_2','lag_3','rolling_4w_mean','rolling_4w_std']]
print(f"Keeping extra columns: {extra_cols}")

df = df.dropna(subset=[TARGET_COL, 'lag_1', 'lag_2', 'lag_3'])
df.to_csv(FEATURE_FILE, index=False)
print(f" Feature engineering complete!")
print(f" Model-ready dataset saved to: {FEATURE_FILE}")
print(f"Shape: {df.shape}")
print(f"Years present: {df['year'].unique()}")




