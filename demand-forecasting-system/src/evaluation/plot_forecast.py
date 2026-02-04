import pandas as pd
import matplotlib.pyplot as plt
import os

CLEANED_DATA_PATH = r'C:\Users\Pranoti munjankar\OneDrive\Desktop\python\retail forecasting\demand-forecasting-system\data\processed\cleaned_sales_data.csv'
FORECAST_PATH = r'C:\Users\Pranoti munjankar\OneDrive\Desktop\python\retail forecasting\demand-forecasting-system\reports\forecasts\forecast_output.csv'
PLOT_OUTPUT_PATH = r'reports/visualizations/forecast_vs_actual.png'

actual = pd.read_csv(CLEANED_DATA_PATH, parse_dates=['date'])
forecast = pd.read_csv(FORECAST_PATH, parse_dates=['date'])


actual.columns = [c.lower() for c in actual.columns]
forecast.columns = [c.lower() for c in forecast.columns]

if 'forecasted_sales' in forecast.columns:
    forecast.rename(columns={'forecasted_sales': 'forecast'}, inplace=True)

df = pd.merge(actual, forecast, on=['store', 'dept', 'date'], how='inner')

agg_df = df.groupby('date')[['weekly_sales', 'forecast']].sum().reset_index()

plt.figure(figsize=(14,6))
plt.plot(agg_df['date'], agg_df['weekly_sales'], label='Actual Sales', color='blue')
plt.plot(agg_df['date'], agg_df['forecast'], label='Forecasted Sales', color='orange')
plt.title('Forecast vs Actual Weekly Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)

os.makedirs(os.path.dirname(PLOT_OUTPUT_PATH), exist_ok=True)
plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
plt.show()

print(f"✅ Forecast vs Actual plot saved to: {PLOT_OUTPUT_PATH}")

