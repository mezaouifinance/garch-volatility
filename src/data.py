import yfinance as yf
import pandas as pd


def load_returns(ticker: str, start: str = "2018-01-01", end: str = None) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    prices = df["Close"].squeeze()
    returns = prices.pct_change().dropna()
    returns.name = ticker
    return returns
