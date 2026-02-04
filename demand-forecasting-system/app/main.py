
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router

app = FastAPI(
    title="Retail Demand Forecasting System",
    description="Demand Forecasting & Inventory Planning API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Retail Demand Forecasting API running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "OK"}


