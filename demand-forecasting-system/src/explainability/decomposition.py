import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')

FEATURE_PATH = 'data/features/sales_features.csv'
MODEL_PATH = 'src/models/xgboost_model.pkl'
EXPLAIN_OUTPUT_PATH = 'reports/visualizations/feature_importance.png'

features = pd.read_csv(FEATURE_PATH)
model = joblib.load(MODEL_PATH)


X = features[['store', 'dept', 'isholiday', 'is_holiday', 'year', 'month', 'day', 'weekday', 'is_weekend',
              'lag_1', 'lag_2', 'lag_3', 'rolling_4w_mean', 'rolling_4w_std']]


importance = model.feature_importances_
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importance})
feat_imp = feat_imp.sort_values(by='importance', ascending=False)


plt.figure(figsize=(10,6))
plt.barh(feat_imp['feature'], feat_imp['importance'])
plt.gca().invert_yaxis()
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
os.makedirs(os.path.dirname(EXPLAIN_OUTPUT_PATH), exist_ok=True)
plt.savefig(EXPLAIN_OUTPUT_PATH)
plt.show()

print("Feature importance plot saved to:", EXPLAIN_OUTPUT_PATH)
