import pandas as pd
import numpy as np
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')


FORECAST_PATH = 'C:\\Users\\Pranoti munjankar\\OneDrive\\Desktop\\python\\retail forecasting\\demand-forecasting-system\\reports\\forecasts\\forecast_output.csv'
CLEANED_DATA_PATH = 'C:\\Users\\Pranoti munjankar\\OneDrive\\Desktop\\python\\retail forecasting\\demand-forecasting-system\\data\\processed\\cleaned_sales_data.csv'

METRICS_OUTPUT_PATH = "reports/metrics/evaluation_metrics.csv"

actual = pd.read_csv(CLEANED_DATA_PATH, parse_dates=['date'])

forecast = pd.read_csv(FORECAST_PATH, parse_dates=['date'])

actual.columns = [c.lower() for c in actual.columns]
forecast.columns = [c.lower() for c in forecast.columns]

if 'forecasted_sales' in forecast.columns:
    forecast.rename(columns={'forecasted_sales': 'forecast'}, inplace=True)

merge_cols = ['store', 'dept', 'date']
for col in merge_cols:
    if col not in actual.columns or col not in forecast.columns:
        raise ValueError(f"Column '{col}' missing in one of the files!")

df = pd.merge(actual, forecast, on=merge_cols, how='inner')


if 'forecast' not in df.columns:
    raise ValueError("Column 'forecast' not found in forecast file!")


y_true = df['weekly_sales'].values
y_pred = df['forecast'].values

rmse = np.sqrt(np.mean((y_true - y_pred)**2))
mae = np.mean(np.abs(y_true - y_pred))
mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100  # avoid div by 0

metrics = pd.DataFrame({
    'RMSE': [rmse],
    'MAE': [mae],
    'MAPE': [mape]
})

os.makedirs(os.path.dirname(METRICS_OUTPUT_PATH), exist_ok=True)
metrics.to_csv(METRICS_OUTPUT_PATH, index=False)


print(" Metrics saved to:", METRICS_OUTPUT_PATH)
print(" Evaluation Metrics:")
print(f"   RMSE: {rmse:.2f}")
print(f"   MAE: {mae:.2f}")
print(f"   MAPE: {mape:.2f}%")
