"""FastAPI prediction service; follows the teacher's sample app.py."""
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_PATH = BASE_DIR / "model" / "model.json"

model = XGBRegressor()
model.load_model(str(MODEL_PATH))  # First run: python train_model.py

app = FastAPI(title="Six-Month Car Sales Forecaster")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

class SalesInput(BaseModel):
    sales: list[float]

@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: SalesInput):
    if len(input.sales) != 6:
        raise HTTPException(status_code=400, detail="Enter exactly 6 months of sales.")
    sales = np.asarray(input.sales, dtype=float)
    if not np.isfinite(sales).all() or (sales < 0).any():
        raise HTTPException(status_code=400, detail="Sales must be finite, nonnegative numbers.")
    # Input order: month -6, -5, -4, -3, -2, -1.
    X_new = sales.reshape(1, -1)
    prediction = float(model.predict(X_new)[0])
    return {"prediction": prediction}
