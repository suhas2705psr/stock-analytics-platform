def generate_signal(row):

    if row["RSI"] < 35 and row["SMA20"] > row["SMA50"]:
        return "BUY"

    if row["RSI"] > 65 and row["SMA20"] < row["SMA50"]:
        return "SELL"

    return "HOLD"
