import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate basic technical indicator features for ML models.
    """

    df["return"] = df["Close"].pct_change()

    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()

    df["volatility_20"] = df["return"].rolling(20).std()

    return df
