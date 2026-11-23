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

## Deployment

Push the full project to GitHub, including `model/model.json`. Create a Render **Web Service** with:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Runtime: Python 3
- Python version: controlled by `.python-version` (`3.13.5`)

## Submission

Provide your GitHub repository link, Render live URL, and screenshot of a successful prediction.

Note: This is a classroom demonstration. The small historic dataset and six-lag model have forecasting limitations.
