import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="AI Stock Analytics",
    layout="wide"
)

st.title("📈 AI-Powered Stock Analytics Platform")


mode = st.sidebar.selectbox(
    "Select Mode",
    [
        "Single Stock Analysis",
        "Portfolio Tracking",
        "Sector Analysis"
    ]
)


# ---------------- DATA FETCH ----------------

def get_stock_data(ticker):

    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        progress=False
    )

    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    return df


# ---------------- INDICATORS ----------------

def add_indicators(df):

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# ---------------- SIGNAL ----------------

def get_signal(rsi):

    if rsi < 30:
        return "BUY"

    if rsi > 70:
        return "SELL"

    return "HOLD"


# ---------------- COLOR STYLE ----------------

def color_signal(val):

    if val == "BUY":
        return "color: #00FF88; font-weight: bold"

    elif val == "SELL":
        return "color: #FF4444; font-weight: bold"

    else:
        return "color: #FFD700; font-weight: bold"


# ---------------- CHART ----------------

def create_chart(df):

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2]
    )

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["SMA20"],
            name="SMA20",
            line=dict(color="blue")
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["SMA50"],
            name="SMA50",
            line=dict(color="orange")
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Volume"
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["RSI"],
            name="RSI",
            line=dict(color="purple")
        ),
        row=3,
        col=1
    )

    fig.add_hline(y=70, row=3, col=1, line_dash="dash", line_color="red")
    fig.add_hline(y=30, row=3, col=1, line_dash="dash", line_color="green")

    fig.update_layout(
        height=800,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    return fig


# ---------------- SINGLE STOCK ----------------

if mode == "Single Stock Analysis":

    ticker = st.text_input("Enter ticker", "AAPL")

    if st.button("Analyze"):

        df = get_stock_data(ticker)
        df = add_indicators(df)

        df["Close"] = df["Close"].squeeze()

        st.subheader(f"{ticker} Analysis")

        fig = create_chart(df)

        st.plotly_chart(fig, use_container_width=True)

        latest_rsi = float(df["RSI"].iloc[-1])
        latest_price = float(df["Close"].iloc[-1])

        signal = get_signal(latest_rsi)

        col1, col2, col3 = st.columns(3)

        col1.metric("Price", round(latest_price, 2))
        col2.metric("RSI", round(latest_rsi, 2))
        col3.metric("Signal", signal)

        st.dataframe(df.tail())


# ---------------- PORTFOLIO ----------------

elif mode == "Portfolio Tracking":

    tickers = st.text_area(
        "Enter tickers separated by comma",
        "AAPL,MSFT,NVDA"
    )

    tickers = [t.strip() for t in tickers.split(",")]

    results = []

    for ticker in tickers:

        df = get_stock_data(ticker)
        df = add_indicators(df)

        rsi = float(df["RSI"].iloc[-1])
        price = float(df["Close"].iloc[-1])

        signal = get_signal(rsi)

        results.append(
            {
                "Ticker": ticker,
                "Price": round(float(df["Close"].iloc[-1]), 2),
                "RSI": round(float(df["RSI"].iloc[-1]), 2),
                "Signal": signal
            }
        )

    portfolio_df = pd.DataFrame(results)

    st.dataframe(
        portfolio_df.style.applymap(color_signal, subset=["Signal"])
    )


# ---------------- SECTOR ----------------

elif mode == "Sector Analysis":

    sectors = {

        "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL"],
        "Banking": ["JPM", "BAC", "C"],
        "Energy": ["XOM", "CVX"],
        "India Banking": ["HDFCBANK.NS", "ICICIBANK.NS"]

    }

    sector = st.selectbox("Select Sector", list(sectors.keys()))

    tickers = sectors[sector]

    results = []

    for ticker in tickers:

        df = get_stock_data(ticker)
        df = add_indicators(df)

        rsi = float(df["RSI"].iloc[-1])
        price = float(df["Close"].iloc[-1])

        signal = get_signal(rsi)

        results.append(
            {
                "Ticker": ticker,
                "Price": round(float(df["Close"].iloc[-1]), 2),
                "RSI": round(float(df["RSI"].iloc[-1]), 2),
                "Signal": signal
            }
        )

    sector_df = pd.DataFrame(results)

    st.dataframe(
        sector_df.style.applymap(color_signal, subset=["Signal"])
    )
