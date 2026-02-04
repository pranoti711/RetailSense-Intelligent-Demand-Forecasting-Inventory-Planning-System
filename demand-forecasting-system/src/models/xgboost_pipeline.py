import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\retail forecasting\demand-forecasting-system"

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURE_PATH = os.path.join(PROJECT_ROOT, "data", "features")
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "models", "xgboost_model.pkl")
FORECAST_PATH = os.path.join(PROJECT_ROOT, "reports", "forecasts", "forecast_output.csv")
EVAL_METRICS_PATH = os.path.join(PROJECT_ROOT, "reports", "metrics", "evaluation_metrics.csv")
PLOT_FORECAST_PATH = os.path.join(PROJECT_ROOT, "reports", "visualizations", "forecast_vs_actual.png")
PLOT_FEATURE_PATH = os.path.join(PROJECT_ROOT, "reports", "visualizations", "feature_importance.png")
INVENTORY_PATH = os.path.join(PROJECT_ROOT, "reports", "inventory", "inventory_plan.csv")

os.makedirs(os.path.dirname(FEATURE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FORECAST_PATH), exist_ok=True)
os.makedirs(os.path.dirname(EVAL_METRICS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PLOT_FORECAST_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PLOT_FEATURE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(INVENTORY_PATH), exist_ok=True)

print("🔹 Step 1: Feature Engineering...")
cleaned_file = os.path.join(PROCESSED_PATH, "cleaned_sales_data.csv")
df = pd.read_csv(cleaned_file)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date").reset_index(drop=True)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = (df['weekday'] >= 5).astype(int)  

TARGET_COL = None
for c in ["weekly_sales", "sales", "total_sales", "units_sold", "quantity"]:
    if c in df.columns:
        TARGET_COL = c
        break
if TARGET_COL is None:
    raise ValueError("No known target column found!")


df['lag_1'] = df[TARGET_COL].shift(1)
df['lag_2'] = df[TARGET_COL].shift(2)
df['lag_3'] = df[TARGET_COL].shift(3)

df['rolling_4w_mean'] = df[TARGET_COL].rolling(window=4, min_periods=1).mean()
df['rolling_4w_std'] = df[TARGET_COL].rolling(window=4, min_periods=1).std().fillna(0)

df = df.dropna(subset=['lag_1','lag_2','lag_3'])


FEATURE_FILE = os.path.join(FEATURE_PATH, "sales_features.csv")
df.to_csv(FEATURE_FILE, index=False)
print(f"✅ Feature engineering complete. Features saved: {FEATURE_FILE}")


print("\n🔹 Step 2: XGBoost Model Training...")
FEATURE_COLS = ['store', 'dept', 'is_holiday', 'year', 'month', 'day', 'weekday', 'is_weekend',
                'lag_1', 'lag_2', 'lag_3', 'rolling_4w_mean', 'rolling_4w_std']

missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing feature columns in data: {missing_cols}. Check pipeline_runner.py for column preparation.")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

train_size = int(0.7 * len(df))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=50)

joblib.dump(model, MODEL_PATH)
print(f"✅ Model trained and saved: {MODEL_PATH}")

print("\n🔹 Step 3: Forecast Generation...")
df['forecast'] = model.predict(X)

df[['store','dept','date','forecast']].to_csv(FORECAST_PATH, index=False)
print(f"✅ Forecast saved: {FORECAST_PATH}")

print("\n🔹 Step 4: Evaluation Metrics...")
y_pred = df['forecast'].values
y_true = df[TARGET_COL].values

rmse = np.sqrt(np.mean((y_true - y_pred)**2))
mae = np.mean(np.abs(y_true - y_pred))
mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0,1,y_true))) * 100

metrics = pd.DataFrame({'RMSE':[rmse],'MAE':[mae],'MAPE':[mape]})
metrics.to_csv(EVAL_METRICS_PATH, index=False)
print(f"✅ Metrics saved: {EVAL_METRICS_PATH}")
print(f"Evaluation Metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

print("\n🔹 Step 5: Forecast vs Actual Plot...")
agg_df = df.groupby('date')[[TARGET_COL,'forecast']].sum().reset_index()
plt.figure(figsize=(14,6))
plt.plot(agg_df['date'], agg_df[TARGET_COL], label='Actual', color='blue')
plt.plot(agg_df['date'], agg_df['forecast'], label='Forecast', color='orange')
plt.title('Forecast vs Actual Weekly Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.savefig(PLOT_FORECAST_PATH, dpi=300)
plt.show()
print(f"✅ Plot saved: {PLOT_FORECAST_PATH}")

print("\n🔹 Step 6: Feature Importance Plot...")
importance = model.feature_importances_
feat_imp = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importance}).sort_values(by='importance', ascending=False)
plt.figure(figsize=(10,6))
plt.barh(feat_imp['feature'], feat_imp['importance'])
plt.gca().invert_yaxis()
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig(PLOT_FEATURE_PATH)
plt.show()
print(f"✅ Feature importance saved: {PLOT_FEATURE_PATH}")

print("\n🔹 Step 7: Inventory Planning...")
df['rolling_mean'] = df.groupby(['store','dept'])['forecast'].transform(lambda x: x.rolling(4,min_periods=1).mean())
df['safety_stock'] = df['rolling_mean'] * 0.2
df['reorder_qty'] = df['forecast'] + df['safety_stock']
df[['store','dept','date','forecast','rolling_mean','safety_stock','reorder_qty']].to_csv(INVENTORY_PATH, index=False)
print(f"✅ Inventory plan saved: {INVENTORY_PATH}")
print(df[['store','dept','date','forecast','rolling_mean','safety_stock','reorder_qty']].head())

print("\n🎯 XGBoost Pipeline Complete!")