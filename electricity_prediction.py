# HOUSEHOLD ELECTRICITY CONSUMPTION PREDICTION
# Data Mining Mini Project - CA2
# Algorithm: Random Forest Regression

import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. CREATE FOLDERS

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 2. DOWNLOAD DATASET

url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"

zip_path = "data/household_power_consumption.zip"
txt_path = "data/household_power_consumption.txt"

if not os.path.exists(txt_path):

    print("Downloading electricity dataset...")

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract("household_power_consumption.txt", "data")

    print("Dataset downloaded successfully!")

# 3. LOAD DATASET

df = pd.read_csv(
    txt_path,
    sep=";",
    na_values="?",
    low_memory=False
)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print(df.head())

# 4. DATA PREPROCESSING

df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

df = df.dropna(
    subset=["datetime", "Global_active_power"]
)

df = df.sort_values("datetime")
df = df.set_index("datetime")

# Fill short gaps in the minute-level readings.
df["Global_active_power"] = (
    df["Global_active_power"]
    .interpolate(limit=5)
)

# Convert minute-level readings into hourly average power.
# For a one-hour interval, average kW is numerically
# equivalent to hourly energy consumption in kWh.

hourly = df["Global_active_power"].resample("h").mean()

hourly = hourly.to_frame(name="consumption_kwh")
hourly = hourly.dropna()

print("\nPreprocessing completed!")
print("Hourly records:", len(hourly))

# 5. FEATURE ENGINEERING

data = hourly.copy()

data["hour"] = data.index.hour
data["day_of_week"] = data.index.dayofweek
data["month"] = data.index.month

data["is_weekend"] = (
    data.index.dayofweek >= 5
).astype(int)

data["lag_1"] = data["consumption_kwh"].shift(1)

data["lag_24"] = data["consumption_kwh"].shift(24)

data["rolling_mean_24"] = (
    data["consumption_kwh"]
    .shift(1)
    .rolling(24)
    .mean()
)

# Predict the next hour's consumption.

data["target"] = data["consumption_kwh"].shift(-1)

data = data.dropna()

features = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "consumption_kwh",
    "lag_1",
    "lag_24",
    "rolling_mean_24"
]

X = data[features]
y = data["target"]

# 6. TRAIN-TEST SPLIT

split = int(len(data) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 7. RANDOM FOREST REGRESSION

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model training completed!")

# 8. MODEL EVALUATION

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\n----- MODEL EVALUATION RESULTS -----")

print("Mean Absolute Error:", round(mae, 4))

print("Mean Squared Error:", round(mse, 4))

print("Root Mean Squared Error:", round(rmse, 4))

print("R2 Score:", round(r2, 4))

# Save evaluation results.

metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],
    "Value": [mae, mse, rmse, r2]
})

metrics.to_csv(
    "results/evaluation_metrics.csv",
    index=False
)

# 9. ACTUAL VS PREDICTED GRAPH

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

comparison.to_csv(
    "results/predictions.csv"
)

plt.figure(figsize=(12, 5))

plt.plot(
    comparison["Actual"].iloc[:168].values,
    label="Actual Consumption"
)

plt.plot(
    comparison["Predicted"].iloc[:168].values,
    label="Predicted Consumption"
)

plt.title("Actual vs Predicted Electricity Consumption")

plt.xlabel("Hours")
plt.ylabel("Electricity Consumption (kWh)")

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/actual_vs_predicted.png"
)

plt.show()

# 10. HOURLY CONSUMPTION GRAPH

hourly_pattern = hourly.groupby(
    hourly.index.hour
)["consumption_kwh"].mean()

plt.figure(figsize=(10, 5))

hourly_pattern.plot(kind="bar")

plt.title("Average Electricity Consumption by Hour")

plt.xlabel("Hour of Day")
plt.ylabel("Average Consumption (kWh)")

plt.tight_layout()

plt.savefig(
    "results/hourly_consumption.png"
)

plt.show()

# 11. FEATURE IMPORTANCE GRAPH

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values()

plt.figure(figsize=(10, 5))

importance.plot(kind="barh")

plt.title("Random Forest Feature Importance")

plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png"
)

plt.show()

# 12. FINAL OUTPUT

print("\n----- PROJECT COMPLETED -----")

print("Algorithm: Random Forest Regression")

print("Predicted electricity consumption successfully.")

print("Graphs and evaluation results saved in results folder.")