RetailSense
Intelligent Demand Forecasting & Inventory Planning System

RetailSense is a production-oriented, modular demand forecasting system designed for retail businesses.
It combines data engineering, machine learning, forecasting, evaluation, and decision support into a single, scalable pipeline with an interactive user interface.

This project demonstrates end-to-end ML system design, from raw data ingestion to actionable inventory insights.

📌 Business Problem Statement

Retail organizations often face:

Overstocking → increased holding costs

Understocking → lost sales and dissatisfied customers

Poor forecasting accuracy due to seasonality and external factors

RetailSense solves this by:

Learning historical demand patterns

Forecasting future demand accurately

Translating forecasts into inventory planning recommendations

🎯 Project Objectives

Build a robust demand forecasting pipeline

Design a clean, modular ML architecture

Enable non-technical users to upload data and view forecasts

Provide quantitative evaluation metrics

Support future scalability and model extensibility

🏛️ System Architecture Overview
User (CSV Upload)
        │
        ▼
Streamlit Dashboard
        │
        ▼
Application Layer (app/)
        │
        ▼
ML Pipeline (src/)
 ├── Preprocessing
 ├── Model Training / Loading
 ├── Forecast Generation
 ├── Evaluation
 └── Inventory Optimization
        │
        ▼
Reports & Visualizations

🗂️ Detailed Folder Structure
app/ – Application Layer

Handles orchestration and communication between UI and ML pipeline.

main.py – Application entry point

pipeline_runner.py – Executes end-to-end ML pipeline

dashboard/ – User Interface

Built with Streamlit for rapid interaction.

CSV upload

Forecast execution

Visualization of predictions vs actuals

Metrics display

src/ – Core ML System
preprocessing/

Missing value handling

Data type normalization

Date parsing

feature_engineering/

Rolling window statistics

Temporal features (day, month, weekday)

Demand trend extraction

models/

XGBoost model definition

Training pipeline

Model persistence

forecasting/

Forecast generation logic

Supports retrained and stored models

evaluation/

RMSE, MAE, MAPE calculation

Forecast vs actual visualization

inventory_optimization/

Converts demand forecasts into inventory plans

Helps avoid stockouts and overstocking

config/

Centralized YAML-based configuration

📊 Data Flow Pipeline

Raw Data Ingestion

Data Cleaning & Validation

Model Training / Loading

Demand Forecasting

Model Evaluation

Inventory Planning

Reporting & Visualization

📁 Dataset Description
Expected Input Schema
Column	Description
OrderID	Unique order identifier
OrderDate	Transaction date
Product	Product name
Category	Product category
Quantity	Units sold
UnitPrice	Price per unit
TotalSales	Quantity × UnitPrice
Region	Sales region

📄 Example file: sample_sales_data.csv

📈 Machine Learning Details
Model Used

XGBoost Regressor

Why XGBoost?

Handles non-linear relationships

Performs well on tabular data

Robust to missing values

Industry-proven algorithm

📏 Evaluation Metrics
Metric	Purpose
RMSE	Penalizes large errors
MAE	Average absolute deviation
MAPE	Relative error interpretation

Metrics are saved to:

reports/metrics/

📊 Visual Outputs

Generated automatically:

Historical sales trends

Forecast vs actual comparison

Feature importance plot

Saved under:

reports/visualizations/

🖥️ Running the Application
Full Pipeline
python run.py

Streamlit Dashboard
streamlit run dashboard/streamlit_app.py

🔐 Configuration Management

All configurable parameters:

File paths

Model parameters

Forecast horizon

Stored in:

src/config/config.yaml

🧪 Experimentation & Notebooks

The notebooks/ directory contains:

Exploratory data analysis

Feature experiments

Model comparison

Error analysis

This supports transparent ML experimentation.

🛠️ Tech Stack

Python 3.10

Pandas, NumPy

Scikit-learn

XGBoost

Streamlit

Matplotlib

Git & GitHub

🧩 Design Principles Followed

Separation of concerns

Modular pipeline design

Reproducibility

Scalability

Maintainability

🔮 Future Enhancements

Multi-model comparison (LSTM, Prophet)

Automated retraining scheduler

REST API deployment

Cloud integration (AWS / GCP)

Role-based dashboard access

CI/CD pipeline

👩‍💼 Author

Pranoti Munjankar
Data Science & Machine Learning Enthusiast

🔗 GitHub: https://github.com/pranoti711

🌟 Final Note

This project demonstrates:

Real-world ML pipeline design

Business-aligned forecasting

Production-ready code structure
