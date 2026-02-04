

import pandas as pd
import joblib
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\retail forecasting\demand-forecasting-system"
FEATURES_FILE = os.path.join(PROJECT_ROOT, "data", "features", "sales_features.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "src", "models", "xgboost_model.pkl")
FORECAST_OUTPUT = os.path.join(PROJECT_ROOT, "reports", "forecasts", "forecast_output.csv")

os.makedirs(os.path.dirname(FORECAST_OUTPUT), exist_ok=True)

print(" Loading feature dataset...")
features = pd.read_csv(FEATURES_FILE)
print(f"Features loaded: {features.shape}")

print(f" Loading trained XGBoost model from {MODEL_FILE}...")
model = joblib.load(MODEL_FILE)

print("🚀 Generating forecasts...")
drop_cols = ['date', 'weekly_sales', 'holiday_name'] 
X = features.drop(columns=drop_cols, errors='ignore')

print(f"Using {X.shape[1]} features: {X.columns.tolist()}")

try:
    features["forecasted_sales"] = model.predict(X)
except ValueError as e:
    print(f"  Feature mismatch: {e}")
    print(f"Expected features by model: {model.n_features_in_}")
    print(f"Got {X.shape[1]} features")
    raise


output_cols = ["store", "dept", "date", "forecasted_sales"]
output_cols = [c for c in output_cols if c in features.columns]
features[output_cols].to_csv(FORECAST_OUTPUT, index=False)
print(f" Forecasts saved to: {FORECAST_OUTPUT}")
print("Forecasting complete!")
