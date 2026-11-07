
import pandas as pd
import numpy as np


def calculate_beta(stock_returns, benchmark_returns):
    """
    Calculate beta using covariance / benchmark variance.
    """

    aligned = pd.concat(
        [stock_returns, benchmark_returns],
        axis=1
    ).dropna()

    if len(aligned) < 2:
        return np.nan

    stock = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]

    benchmark_variance = benchmark.var()

    if benchmark_variance == 0:
        return np.nan

    beta = stock.cov(benchmark) / benchmark_variance

    return beta


def calculate_sharpe_ratio(returns, risk_free_rate=0.06):
    """
    Calculate annualized Sharpe Ratio.

    risk_free_rate is an annual decimal rate.
    """

    daily_risk_free = (
        (1 + risk_free_rate) ** (1 / 252) - 1
    )

    excess_returns = returns - daily_risk_free

    daily_std = excess_returns.std()

    if daily_std == 0 or pd.isna(daily_std):
        return np.nan

    return (
        excess_returns.mean() / daily_std
    ) * np.sqrt(252)


def calculate_var(returns, confidence=0.95):
    """
    Historical Value at Risk.

    Returns a positive loss percentage.
    """

    quantile = returns.quantile(1 - confidence)

    return max(0.0, -quantile) * 100


def calculate_cvar(returns, confidence=0.95):
    """
    Historical Conditional Value at Risk.
    """

    var_threshold = returns.quantile(1 - confidence)

    tail_returns = returns[returns <= var_threshold]

    if tail_returns.empty:
        return np.nan

    return max(0.0, -tail_returns.mean()) * 100


def calculate_max_drawdown(prices):
    """
    Calculate maximum peak-to-trough drawdown.
    """

    if prices.empty:
        return np.nan

    running_max = prices.cummax()

    drawdown = (prices / running_max) - 1

    return drawdown.min() * 100


def calculate_cagr(prices):
    """
    Calculate realized compounded annual growth rate
    using the number of calendar days between observations.
    """

    prices = prices.dropna()

    if len(prices) < 2:
        return np.nan

    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]

    elapsed_days = (
        prices.index[-1] - prices.index[0]
    ).days

    if start_price <= 0 or elapsed_days <= 0:
        return np.nan

    return (
        (end_price / start_price) **
        (365.25 / elapsed_days) - 1
    ) * 100


def calculate_risk_metrics(
    prices,
    returns,
    benchmark,
    risk_free_rate=0.06
):
    """
    Calculate risk metrics for every stock.
    """

    results = {}

    for ticker in returns.columns:

        stock_returns = returns[ticker].dropna()
        stock_prices = prices[ticker].dropna()

        results[ticker] = {
            "Beta": calculate_beta(
                stock_returns,
                benchmark
            ),

            "Sharpe Ratio": calculate_sharpe_ratio(
                stock_returns,
                risk_free_rate
            ),

            "Annualized Volatility (%)": (
                stock_returns.std() * np.sqrt(252) * 100
            ),

            "95% VaR (%)": calculate_var(
                stock_returns
            ),

            "95% CVaR (%)": calculate_cvar(
                stock_returns
            ),

            "Maximum Drawdown (%)": calculate_max_drawdown(
                stock_prices
            ),

            "Realized CAGR (%)": calculate_cagr(
                stock_prices
            )
        }

    return pd.DataFrame(results).T.round(4)
