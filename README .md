# Stock Analytics Platform

An end-to-end stock analytics system for exploring financial market data using
technical indicators, machine learning models, and interactive dashboards.

Built using Python, Streamlit, and real market data from Yahoo Finance.

---

## What it does

The dashboard has four modes:

**Single Stock Analysis** — enter any ticker and get a candlestick chart with moving averages, a volume panel, RSI, and MACD. Works for any market, not just US stocks.

**Portfolio Tracking** — paste in a list of tickers and get a summary table showing price, daily change, RSI, and a buy/sell/hold signal for each one.

**Sector Analysis** — pick a sector (US Tech, India Banking, European Auto, etc.) and compare all the stocks in it side by side.

**Market Screener** — scans a universe of stocks across US, India, and Europe and filters by signal strength. Useful for quickly finding oversold or overbought conditions.

---

## Why I built it

Most tutorials use toy datasets. I wanted to work with real data that actually changes every day — prices, volume, market conditions — and build something that could handle edge cases like missing data, different currencies, and markets that open and close at different times.

The project also gave me a reason to go deep on time-series concepts like indicator lag, lookback windows, and why backtesting on historical data can be misleading if done carelessly.

---


## Stack

- **Data**: yfinance (Yahoo Finance API), Pandas, PyArrow
- **Indicators**: custom implementation in NumPy/Pandas — RSI, MACD, SMA, EMA, Bollinger Bands
- **ML**: scikit-learn for baseline models, TensorFlow/Keras for LSTM sequence modeling
- **Dashboard**: Streamlit + Plotly
- **Tests**: pytest

---

## Project layout

```
stock-analytics/
├── src/
│   ├── analytics/
│   │   ├── indicators.py       # all indicator calculations
│   │   └── signals.py          # signal scoring logic
│   ├── clients/
│   │   └── yahoo_client.py     # data fetching and caching
│   ├── dashboard/
│   │   └── streamlit_app.py    # the main app
│   ├── models/
│   │   ├── ml_model.py
│   │   ├── tf_model.py
│   │   └── ensemble.py
│   └── pipeline/
│       ├── ingest.py
│       ├── transform.py
│       ├── features.py
│       ├── backtest.py
│       ├── compare.py
│       └── viz.py
├── tests/
│   └── test_pipeline.py
├── data/               # gitignored
├── requirements.txt
└── README.md
```

---

## Running it

```bash
git clone https://github.com/YOUR_USERNAME/stock-analytics.git
cd stock-analytics

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
streamlit run src/dashboard/streamlit_app.py
```

---

## Supported markets

The ticker system follows Yahoo Finance conventions:

- US stocks — `AAPL`, `NVDA`, `TSLA`
- India NSE — `HDFCBANK.NS`, `ICICIBANK.NS`, `RELIANCE.NS`
- Germany — `BMW.DE`, `SAP.DE`
- UK — `SHELL.L`, `HSBA.L`

---

## On the ML side

The LSTM model treats price prediction as a sequence problem — given the last N days, predict direction for day N+1. It's not a magic money-printer. In most backtests, simple buy-and-hold beats it over long bull runs, which is itself a useful finding.

The ensemble combines LSTM output with the scikit-learn classifier to reduce variance. Signals are benchmarked against a baseline strategy so the numbers are honest.

---


## Key Insights

• Simple rule-based strategies can outperform complex models in certain regimes  
• LSTM captures temporal patterns but is sensitive to market regime shifts  
• Proper backtesting is critical to avoid lookahead bias  

---


## Tests

```bash
pytest tests/
```

---

## Notes

Data files are gitignored — the pipeline fetches fresh data on first run and caches locally. If a ticker returns no data, it's usually a suffix issue (check Yahoo Finance for the correct format for your market).
