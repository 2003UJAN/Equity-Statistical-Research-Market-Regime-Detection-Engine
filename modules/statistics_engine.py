
import pandas as pd
import numpy as np


def calculate_descriptive_statistics(returns):
    """
    Calculate descriptive statistics for daily stock returns.
    """

    stats = pd.DataFrame(index=returns.columns)

    stats["Mean Daily Return (%)"] = returns.mean() * 100

    stats["Annualized Return (%)"] = (
        (1 + returns.mean()) ** 252 - 1
    ) * 100

    stats["Variance"] = returns.var()

    stats["Daily Std Dev (%)"] = returns.std() * 100

    stats["Annualized Volatility (%)"] = (
        returns.std() * np.sqrt(252) * 100
    )

    stats["Skewness"] = returns.skew()

    stats["Kurtosis"] = returns.kurtosis()

    stats["Minimum Return (%)"] = returns.min() * 100

    stats["Maximum Return (%)"] = returns.max() * 100

    stats["Observations"] = returns.count()

    return stats.round(4)


def calculate_correlation(returns):
    """
    Calculate Pearson correlation matrix.
    """

    return returns.corr(method="pearson")


def calculate_covariance(returns):
    """
    Calculate daily covariance matrix.
    """

    return returns.cov()


def calculate_rolling_volatility(returns, window=20):
    """
    Calculate annualized rolling volatility.
    """

    return returns.rolling(window).std() * np.sqrt(252) * 100


def calculate_rolling_correlation(
    returns,
    stock,
    benchmark,
    window=60
):
    """
    Calculate rolling correlation between a stock and benchmark.
    """

    if stock not in returns.columns or benchmark not in returns.columns:
        return pd.Series(dtype=float)

    return returns[stock].rolling(window).corr(
        returns[benchmark]
    )
