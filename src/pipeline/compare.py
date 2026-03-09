# src/compare.py

import pandas as pd
from pathlib import Path
from src.backtest import generate_signals, backtest_strategy, backtest_ml_strategy
from src.models.ml_model import prepare_dataset, train_logistic_regression

PROCESSED_DIR = Path("data/processed")


if __name__ == "__main__":
    ticker = "AAPL"
    df = pd.read_parquet(PROCESSED_DIR / f"{ticker}.parquet")

    # Rule-based strategy
    rule_df = generate_signals(df)
    rule_df = backtest_strategy(rule_df)

    # ML strategy
    X, y = prepare_dataset(df)
    model = train_logistic_regression(X, y)
    ml_df = backtest_ml_strategy(df, model)

    print("Rule-based final return:", rule_df["cumulative_return"].iloc[-1])
    print("ML-based final return:", ml_df["ml_cumulative_return"].iloc[-1])
