
import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=900, show_spinner=False)
def download_stock_data(tickers, period="2y", interval="1d"):
    """
    Download adjusted historical closing prices for multiple tickers.

    Returns:
        DataFrame: Date-indexed prices, one column per ticker.
    """

    if not tickers:
        return pd.DataFrame()

    price_data = {}

    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if data.empty:
                continue

            # Handle both ordinary and MultiIndex Yahoo Finance output
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]

                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            else:
                close = data["Close"]

            close = pd.to_numeric(close, errors="coerce")
            close.name = ticker

            price_data[ticker] = close

        except Exception:
            continue

    if not price_data:
        return pd.DataFrame()

    prices = pd.concat(price_data.values(), axis=1)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    return prices


def prepare_market_data(prices):
    """
    Clean prices and calculate daily percentage returns.
    """

    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    prices = prices.copy()

    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(how="all")

    # Forward-fill only short missing observations.
    prices = prices.ffill(limit=3)

    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([float("inf"), float("-inf")], pd.NA)
    returns = returns.dropna(how="all")

    return prices, returns
