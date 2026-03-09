# src/models/tf_model.py

import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("data/processed")


def create_sequences(data, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(int(data[i + window] > 0))
    return np.array(X), np.array(y)


def prepare_tf_data(df):
    returns = df["daily_return"].values
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns.reshape(-1, 1)).flatten()

    return create_sequences(returns_scaled)


def train_lstm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], 1)),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train[..., np.newaxis],
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test[..., np.newaxis], y_test),
        verbose=1
    )

    return model


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DIR / "AAPL.parquet")
    X, y = prepare_tf_data(df)
    model = train_lstm(X, y)
