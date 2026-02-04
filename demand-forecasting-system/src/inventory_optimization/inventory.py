import pandas as pd
import numpy as np
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

FORECAST_PATH = r'C:\Users\Pranoti munjankar\OneDrive\Desktop\python\retail forecasting\demand-forecasting-system\reports\forecasts\forecast_output.csv'
INVENTORY_OUTPUT_PATH = r'C:\Users\Pranoti munjankar\OneDrive\Desktop\python\retail forecasting\demand-forecasting-system\reports\inventory\inventory_plan.csv'

forecast = pd.read_csv(FORECAST_PATH, parse_dates=['date'])

forecast.columns = [c.lower() for c in forecast.columns]

if 'forecasted_sales' in forecast.columns:
    forecast.rename(columns={'forecasted_sales': 'forecast'}, inplace=True)
elif 'forecast' not in forecast.columns:
    raise ValueError("Forecast column not found in forecast file!")

forecast['rolling_mean'] = forecast.groupby(['store', 'dept'])['forecast'].transform(
    lambda x: x.rolling(4, min_periods=1).mean()
)


forecast['safety_stock'] = forecast['rolling_mean'] * 0.2

forecast['reorder_qty'] = forecast['forecast'] + forecast['safety_stock']


os.makedirs(os.path.dirname(INVENTORY_OUTPUT_PATH), exist_ok=True)
forecast.to_csv(INVENTORY_OUTPUT_PATH, index=False)

print(" Inventory plan saved to:", INVENTORY_OUTPUT_PATH)
print("Sample inventory plan:")
print(forecast[['store', 'dept', 'date', 'forecast', 'rolling_mean', 'safety_stock', 'reorder_qty']].head())
