import pandas as pd

# -------------------------------
# STEP 1: Load Phase-1 Data
# -------------------------------
df = pd.read_csv("phase1_features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# -------------------------------
# STEP 2: Price-Based Features
# -------------------------------
df["Return_1d"] = df["Close"].pct_change()

df["Close_lag_1"] = df["Close"].shift(1)
df["Close_lag_3"] = df["Close"].shift(3)
df["Close_lag_5"] = df["Close"].shift(5)

df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_10"] = df["Close"].rolling(window=10).mean()

# -------------------------------
# STEP 3: Sentiment Lag Features
# -------------------------------
df["Trend_lag_1"] = df["Search_Interest"].shift(1)
df["Trend_lag_3"] = df["Search_Interest"].shift(3)

# -------------------------------
# STEP 4: Drop Initial NaNs
# -------------------------------
df = df.dropna().reset_index(drop=True)

# -------------------------------
# STEP 5: Save Phase-2 Feature Set
# -------------------------------
OUTPUT_FILE = "phase2_features.csv"
df.to_csv(OUTPUT_FILE, index=False)

print("Phase 2 Feature Engineering Completed")
print("Saved:", OUTPUT_FILE)
print(df.head())
