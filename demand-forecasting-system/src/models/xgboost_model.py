
import pandas as pd
import os
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROJECT_ROOT = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\retail forecasting\demand-forecasting-system"
FEATURE_FILE = os.path.join(PROJECT_ROOT, "data", "features", "sales_features.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "models", "xgboost_model.pkl")
PREDICTIONS_FILE = os.path.join(PROJECT_ROOT, "reports", "forecasts", "xgb_predictions.csv")

os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)

print("🚀 Starting XGBoost model training...")

df = pd.read_csv(FEATURE_FILE)
df['date'] = pd.to_datetime(df['date'])


TARGET_COL = None
for c in ["weekly_sales", "sales", "total_sales", "units_sold", "quantity"]:
    if c in df.columns:
        TARGET_COL = c
        break
if TARGET_COL is None:
    raise ValueError("No known target column found!")

drop_cols = ['date', TARGET_COL, 'holiday_name'] if 'holiday_name' in df.columns else ['date', TARGET_COL]
X = df.drop(columns=drop_cols)
y = df[TARGET_COL]

print(f"Features used: {X.columns.tolist()}")
print(f"Target: {TARGET_COL}")

split_year = df['date'].dt.year.max() 
train_idx = df['date'].dt.year < split_year
test_idx = df['date'].dt.year == split_year

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Training rows: {X_train.shape[0]}, Testing rows: {X_test.shape[0]}")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,
    verbose=50
)

joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved to: {MODEL_PATH}")


y_pred = model.predict(X_test)
pred_df = X_test.copy()
pred_df[TARGET_COL + "_pred"] = y_pred
pred_df['date'] = df.loc[test_idx, 'date'].values
pred_df.to_csv(PREDICTIONS_FILE, index=False)
print(f"📦 Predictions saved to: {PREDICTIONS_FILE}")

rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
print(f"📊 Test RMSE: {rmse:.2f}, MAE: {mae:.2f}")
