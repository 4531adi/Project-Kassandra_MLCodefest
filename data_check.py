
import yfinance as yf
import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import sys

STOCK_NAME = "TSLA"
KEYWORD = "Tesla stock"

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365)

print(f"Fetching data from: {START_DATE.date()} to {END_DATE.date()}")

print("\nFetching stock price data...")

try:
    stock_data = yf.download(
        STOCK_NAME,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        threads=False
    )
except Exception as e:
    print("Stock download failed:", e)
    sys.exit(1)

if stock_data.empty:
    print("ERROR: Stock data is empty (Yahoo rate limit).")
    print("Please wait 10–15 minutes and re-run the script.")
    sys.exit(1)

stock_data = stock_data.reset_index()


if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)

stock_data = stock_data[['Date', 'Close']]

print("Stock data fetched successfully.")
print(stock_data.head())


print("\nFetching Google Trends data...")

pytrends = TrendReq(hl='en-US', tz=360)

pytrends.build_payload(
    kw_list=[KEYWORD],
    timeframe=f"{START_DATE.date()} {END_DATE.date()}",
    geo=""
)

trends_data = pytrends.interest_over_time()

if trends_data.empty:
    print("ERROR: Google Trends data is empty.")
    sys.exit(1)


trends_data = trends_data.reset_index()
trends_data = trends_data[['date', KEYWORD]]
trends_data.columns = ['Date', 'Search_Interest']


trends_data = (
    trends_data
    .set_index('Date')
    .resample('D')
    .ffill()
    .reset_index()
)

print("Google Trends data fetched and expanded to daily.")
print(trends_data.head())

print("\nMerging datasets...")

merged_data = pd.merge(
    stock_data,
    trends_data,
    on="Date",
    how="left"
)


merged_data['Search_Interest'] = (
    merged_data['Search_Interest']
    .ffill()
    .bfill()
)

print("Datasets merged successfully.")
print(merged_data.head())


OUTPUT_FILE = "phase1_features.csv"
merged_data.to_csv(OUTPUT_FILE, index=False)

print("\nPhase 1 completed successfully!")
print("Output saved as:", OUTPUT_FILE)
print("Last 5 rows:")
print(merged_data.tail())
