# src/ingest.py

import os
from dotenv import load_dotenv
from src.clients.yahoo_client import fetch_stock_data, save_raw_data

load_dotenv()


def ingest_tickers(tickers: list[str]) -> None:
    start_date = os.getenv("DEFAULT_START_DATE", "2018-01-01")
    interval = os.getenv("DEFAULT_INTERVAL", "1d")

    for ticker in tickers:
        try:
            print(f"[INFO] Fetching data for {ticker}")
            df = fetch_stock_data(
                ticker=ticker,
                start_date=start_date,
                interval=interval
            )
            path = save_raw_data(df, ticker)
            print(f"[SUCCESS] Saved {ticker} data → {path}")
        except Exception as e:
            print(f"[ERROR] Failed for {ticker}: {e}")


if __name__ == "__main__":
    ingest_tickers(["AAPL", "MSFT", "GOOGL"])
