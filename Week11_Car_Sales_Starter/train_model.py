"""Train and export a six-lag XGBoost car-sales forecasting model."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "monthly-car-sales.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# 1. Read and check the monthly time series.
df = pd.read_csv(DATA_PATH)
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)
if df[["Month", "Sales"]].isna().any().any():
    raise ValueError("Missing Month or Sales values found.")
if df["Month"].duplicated().any():
    raise ValueError("Duplicate months found.")
expected = pd.date_range(df["Month"].iloc[0], df["Month"].iloc[-1], freq="MS")
if len(expected) != len(df) or not (df["Month"].to_numpy() == expected.to_numpy()).all():
    raise ValueError("The dataset must contain consecutive months.")

# 2. lag_6 is the oldest; lag_1 is the most recent.
for lag in range(1, 7):
    df[f"lag_{lag}"] = df["Sales"].shift(lag)
features = [f"lag_{lag}" for lag in range(6, 0, -1)]
ready = df.dropna().reset_index(drop=True)
X = ready[features].to_numpy(dtype=float)
y = ready["Sales"].to_numpy(dtype=float)

# 3. Evaluate on 1968 only; never shuffle chronological observations.
X_train, X_test = X[:-12], X[-12:]
y_train, y_test = y[:-12], y[-12:]
params = dict(n_estimators=150, max_depth=2, learning_rate=0.05,
              objective="reg:squarederror", random_state=42, n_jobs=2)
model = XGBRegressor(**params)
model.fit(X_train, y_train)
test_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
naive_pred = X_test[:, -1]  # Baseline: next month = previous month.
naive_mae = mean_absolute_error(y_test, naive_pred)
print(f"Original months: {len(df)}, usable training examples: {len(ready)}")
print(f"Train examples: {len(X_train)}; test examples: {len(X_test)}")
print(f"1968 XGBoost MAE: {mae:,.2f}; RMSE: {rmse:,.2f}")
print(f"1968 naive baseline MAE: {naive_mae:,.2f}")

# 4. Retrain on all known months before exporting the deployment model.
final_model = XGBRegressor(**params)
final_model.fit(X, y)
model_path = MODEL_DIR / "model.json"
final_model.save_model(str(model_path))
print(f"Saved production model: {model_path}")

# 5. Forecast the month after the final observation.
last_six = df["Sales"].tail(6).to_numpy(dtype=float).reshape(1, -1)
next_month = (df["Month"].iloc[-1] + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
forecast = float(final_model.predict(last_six)[0])
print(f"Input (oldest -> newest): {last_six.flatten().astype(int).tolist()}")
print(f"Forecast for {next_month}: {forecast:,.2f} car sales")
