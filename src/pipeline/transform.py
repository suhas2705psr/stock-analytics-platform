# src/transform.py

from pathlib import Path
import pandas as pd

# Directories
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_stock_data(ticker: str) -> pd.DataFrame:
    """
    Clean and normalize raw stock data for a single ticker.
    Handles Yahoo Finance quirks such as stringified tuple columns.
    """
    file_path = RAW_DIR / f"{ticker}.parquet"
    df = pd.read_parquet(file_path)

    # --- Normalize column names ---
    normalized_cols = []
    for col in df.columns:
        # Handles columns like "('Date', '')" or "('Close', 'AAPL')"
        if isinstance(col, str) and col.startswith("("):
            try:
                col = eval(col)[0]
            except Exception:
                pass
        normalized_cols.append(col)

    df.columns = normalized_cols

    # --- Validate Date column ---
    if "Date" not in df.columns:
        raise ValueError(
            f"Date column not found after normalization. Columns: {df.columns.tolist()}"
        )

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # --- Time-series safe cleaning ---
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset="Date")
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic financial features for analysis.
    """
    df["daily_return"] = df["Close"].pct_change()
    df["sma_20"] = df["Close"].rolling(window=20).mean()
    df["sma_50"] = df["Close"].rolling(window=50).mean()

    df = df.dropna().reset_index(drop=True)
    return df


def save_processed_data(df: pd.DataFrame, ticker: str) -> Path:
    """
    Save processed stock data.
    """
    output_path = PROCESSED_DIR / f"{ticker}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOGL"]

    for symbol in symbols:
        cleaned_df = clean_stock_data(symbol)
        featured_df = add_features(cleaned_df)
        path = save_processed_data(featured_df, symbol)
        print(f"[SUCCESS] Cleaned & featured data saved → {path}")
