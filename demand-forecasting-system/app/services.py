import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

XGBOOST_SCRIPT = os.path.join(
    BASE_DIR, "src", "models", "xgboost_pipeline.py"
)


def run_forecast_pipeline():
    """
    Runs the XGBoost pipeline as a subprocess
    """
    try:
        subprocess.run(
            [sys.executable, XGBOOST_SCRIPT],
            check=True
        )

        return {
            "status": "success",
            "message": "Forecast generated successfully",
            "outputs": {
                "forecast": "reports/forecasts/forecast_output.csv",
                "metrics": "reports/metrics/evaluation_metrics.csv",
                "inventory": "reports/inventory/inventory_plan.csv",
                "plots": "reports/visualizations/"
            }
        }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": str(e)
        }
