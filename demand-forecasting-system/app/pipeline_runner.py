import pandas as pd
import os
import subprocess

UPLOAD_FILE = "app/uploads/latest_upload.csv"
PROCESSED_FILE = "data/processed/cleaned_sales_data.csv"

FEATURE_COLS = [
    'store', 'dept', 'is_holiday', 'year', 'month', 'day',
    'weekday', 'is_weekend', 'lag_1', 'lag_2', 'lag_3',
    'rolling_4w_mean', 'rolling_4w_std'
]
REQUIRED_TARGETS = ['sales', 'quantity', 'total_sales']


COLUMN_MAPPING = {
    'OrderDate': 'date',
    'Region': 'store',
    'Product': 'dept',
    'TotalSales': 'sales',
    'Quantity': 'quantity'
}

def run_pipeline():
    if not os.path.exists(UPLOAD_FILE):
        raise FileNotFoundError("No CSV uploaded.")

    try:
        
        df = pd.read_csv(UPLOAD_FILE)

       
        df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})

        
        for col in FEATURE_COLS + ['date']:
            if col not in df.columns:
                if col in ['store', 'dept']:
                    df[col] = 0  
                elif col == 'is_holiday':
                    df[col] = 0  
                else:
                    df[col] = 0  

        
        TARGET_COL = None
        for t in REQUIRED_TARGETS:
            if t in df.columns:
                TARGET_COL = t
                break
        if TARGET_COL is None:
            df['sales'] = 0
            TARGET_COL = 'sales'

        
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0)
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

       
        df = df.sort_values('date').reset_index(drop=True)

        for lag in ['lag_1', 'lag_2', 'lag_3']:
            lag_num = int(lag.split('_')[1])
            df[lag] = df[TARGET_COL].shift(lag_num).fillna(0)

        window_size = min(28, len(df) // 2) if len(df) > 1 else 1
        df['rolling_4w_mean'] = df[TARGET_COL].rolling(window=window_size).mean().fillna(0)
        df['rolling_4w_std'] = df[TARGET_COL].rolling(window=window_size).std().fillna(0)

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

        for col in ['store', 'dept']:
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col])[0]

        print(f"Processed DataFrame columns: {list(df.columns)}")
        print(f"Expected FEATURE_COLS: {FEATURE_COLS}")
        missing = [col for col in FEATURE_COLS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in processed data: {missing}")

        os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
        df.to_csv(PROCESSED_FILE, index=False)

        result = subprocess.run([
            "python",
            "src/models/xgboost_pipeline.py"
        ], capture_output=True, text=True, check=True)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Pipeline failed: XGBoost script error: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {str(e)}")