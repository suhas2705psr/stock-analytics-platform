# src/clients/yahoo_client.py

from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf


RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_stock_data(
    ticker: str,
    start_date: str = "2018-01-01",
    interval: str = "1d"
) -> pd.DataFrame:

    df = yf.download(
        ticker,
        start=start_date,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    df.reset_index(inplace=True)
    df["ticker"] = ticker
    df["ingested_at"] = datetime.utcnow()

    return df


def save_raw_data(df: pd.DataFrame, ticker: str) -> Path:

    output_path = RAW_DATA_DIR / f"{ticker}.parquet"
    df.to_parquet(output_path, index=False)

    return output_path
