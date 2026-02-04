
from fastapi import APIRouter, UploadFile, File
import os
import shutil
from app.pipeline_runner import run_pipeline
import pandas as pd


router = APIRouter(prefix="/api", tags=["Forecasting"])

UPLOAD_DIR = "app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"error": "Only CSV files are allowed"}

    saved_file = os.path.join(UPLOAD_DIR, "latest_upload.csv")
    with open(saved_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": f"CSV uploaded successfully as {saved_file}"}

@router.post("/run-pipeline")
def run_forecast():
    try:
        run_pipeline()
        return {"status": "success", "message": "Pipeline executed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/forecast")
def get_forecast():
    forecast_path = "reports/forecasts/forecast_output.csv"
    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)
        if len(df) == 0:
            return {"error": "Forecast file is empty"}
        return df.head(100).to_dict(orient="records")
    return {"error": "Forecast not available"}

@router.get("/inventory")
def get_inventory():
    inventory_path = "reports/inventory/inventory_plan.csv"
    if os.path.exists(inventory_path):
        df = pd.read_csv(inventory_path)
        if len(df) == 0:
            return {"error": "Inventory file is empty"}
        return df.head(100).to_dict(orient="records")
    return {"error": "Inventory not available"}



