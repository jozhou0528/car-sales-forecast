# Week 11: Car Sales Forecasting with FastAPI

**Data:** Monthly car sales, January 1960–December 1968 (108 observations).
**Method:** XGBRegressor with six chronological lag inputs (oldest → newest).

## Setup

```bash
python -m pip install -r requirements.txt
python train_model.py
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000/>. Click **Fill 1968 Example** to use the real last six months of data.


