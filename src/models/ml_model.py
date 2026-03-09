# src/models/ml_model.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

PROCESSED_DIR = Path("data/processed")


def prepare_dataset(df: pd.DataFrame):
    """
    Prepare features and target for ML.
    Target: 1 if next-day return is positive, else 0
    """
    df = df.copy()

    df["target"] = (df["daily_return"].shift(-1) > 0).astype(int)
    df = df.dropna()

    features = df[["daily_return", "sma_20", "sma_50"]]
    target = df["target"]

    return features, target


def train_logistic_regression(X, y):
    """
    Train logistic regression classifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model

def generate_signal(row):

    if row["RSI"] < 30:
        return "BUY"

    if row["RSI"] > 70:
        return "SELL"

    return "HOLD"


if __name__ == "__main__":
    ticker = "AAPL"
    file_path = PROCESSED_DIR / f"{ticker}.parquet"

    df = pd.read_parquet(file_path)

    X, y = prepare_dataset(df)
    model = train_logistic_regression(X, y)
