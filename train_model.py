import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# STEP 1: Load Feature Data
# -------------------------------
df = pd.read_csv("phase2_features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# -------------------------------
# STEP 2: Define Target
# -------------------------------
df["Target_Close_Next_Day"] = df["Close"].shift(-1)
df = df.dropna().reset_index(drop=True)

FEATURE_COLS = [
    "Return_1d",
    "Close_lag_1",
    "Close_lag_3",
    "Close_lag_5",
    "MA_5",
    "MA_10",
    "Trend_lag_1",
    "Trend_lag_3"
]

X = df[FEATURE_COLS]
y = df["Target_Close_Next_Day"]

# -------------------------------
# STEP 3: Time-Based Split
# -------------------------------
split_index = int(len(df) * 0.8)

X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

val_dates = df.iloc[split_index:]["Date"]
val_actual = df.iloc[split_index:]["Close"]

# -------------------------------
# STEP 4: Build ML Pipeline
# -------------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

model.fit(X_train, y_train)

# -------------------------------
# STEP 5: Predictions
# -------------------------------
val_predictions = model.predict(X_val)

# -------------------------------
# STEP 6: Evaluation Metrics
# -------------------------------
mae = mean_absolute_error(y_val, val_predictions)
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

print("Validation MAE:", round(mae, 2))
print("Validation RMSE:", round(rmse, 2))

# -------------------------------
# STEP 7: Save Prediction Log
# -------------------------------
prediction_log = pd.DataFrame({
    "Date": val_dates.values,
    "Actual_Closing_Price": val_actual.values,
    "Predicted_Closing_Price": val_predictions
})

prediction_log.to_csv("prediction_log.csv", index=False)

print("Prediction log saved: prediction_log.csv")
print(prediction_log.head())
