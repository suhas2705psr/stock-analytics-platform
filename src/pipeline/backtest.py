# src/backtest.py

from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy/sell signals using SMA crossover strategy.
    """
    df = df.copy()
    df["signal"] = 0

    df.loc[df["sma_20"] > df["sma_50"], "signal"] = 1
    df.loc[df["sma_20"] < df["sma_50"], "signal"] = -1

    return df


def backtest_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backtest strategy performance.
    """
    df = df.copy()

    df["strategy_return"] = df["signal"].shift(1) * df["daily_return"]
    df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()

    return df

def backtest_ml_strategy(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Backtest ML-based trading strategy.
    """
    df = df.copy()

    features = df[["daily_return", "sma_20", "sma_50"]]
    df["ml_signal"] = model.predict(features)

    # Convert prediction to trading position
    df["ml_position"] = df["ml_signal"].replace({0: -1, 1: 1})

    df["ml_strategy_return"] = df["ml_position"].shift(1) * df["daily_return"]
    df["ml_cumulative_return"] = (1 + df["ml_strategy_return"]).cumprod()

    return df



if __name__ == "__main__":
    ticker = "AAPL"
    file_path = PROCESSED_DIR / f"{ticker}.parquet"

    df = pd.read_parquet(file_path)

    df = generate_signals(df)
    df = backtest_strategy(df)

    print(df[["Date", "signal", "strategy_return"]].tail())
    print(f"Final cumulative return: {df['cumulative_return'].iloc[-1]:.2f}")
