# src/viz.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Directory where processed data is stored
PROCESSED_DIR = Path("data/processed")


def plot_price_with_moving_averages(ticker: str):
    """
    Plot closing price along with 20-day and 50-day moving averages.
    Used for trend and momentum analysis.
    """
    file_path = PROCESSED_DIR / f"{ticker}.parquet"
    df = pd.read_parquet(file_path)

    plt.figure(figsize=(12, 6))

    plt.plot(df["Date"], df["Close"], label="Close Price")
    plt.plot(df["Date"], df["sma_20"], label="SMA 20")
    plt.plot(df["Date"], df["sma_50"], label="SMA 50")

    plt.title(f"{ticker} Stock Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_daily_returns(ticker: str):
    """
    Plot daily returns to visualize volatility.
    """
    file_path = PROCESSED_DIR / f"{ticker}.parquet"
    df = pd.read_parquet(file_path)

    plt.figure(figsize=(12, 4))
    plt.plot(df["Date"], df["daily_return"], label="Daily Return")

    plt.axhline(0)
    plt.title(f"{ticker} Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_strategy_performance(ticker: str):
    """
    Plot cumulative strategy returns over time.
    """
    from src.backtest import generate_signals, backtest_strategy

    file_path = PROCESSED_DIR / f"{ticker}.parquet"
    df = pd.read_parquet(file_path)

    df = generate_signals(df)
    df = backtest_strategy(df)

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["cumulative_return"], label="Strategy Return")

    plt.title(f"{ticker} Strategy Performance (SMA Crossover)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    SYMBOL = "AAPL"

    plot_price_with_moving_averages(SYMBOL)
    plot_daily_returns(SYMBOL)
    plot_strategy_performance(SYMBOL)
