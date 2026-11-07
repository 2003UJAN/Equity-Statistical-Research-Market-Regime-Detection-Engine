
import numpy as np
import pandas as pd

from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller, acf, pacf
from arch import arch_model


# ==================================================
# 1. CAPM REGRESSION
# ==================================================

def capm_regression(
    stock_returns,
    market_returns,
    risk_free_rate=0.06
):
    """
    Estimate CAPM using OLS regression.

    Model:
        Ri - Rf = Alpha + Beta * (Rm - Rf) + Error

    Parameters:
        stock_returns: Daily stock returns
        market_returns: Daily benchmark returns
        risk_free_rate: Annual risk-free rate (decimal)

    Returns:
        Dictionary containing CAPM regression results.
    """

    daily_rf = (
        (1 + risk_free_rate) ** (1 / 252) - 1
    )

    aligned = pd.concat(
        [stock_returns, market_returns],
        axis=1
    ).dropna()

    if len(aligned) < 30:
        return None

    stock_excess = aligned.iloc[:, 0] - daily_rf

    market_excess = aligned.iloc[:, 1] - daily_rf

    result = linregress(
        market_excess,
        stock_excess
    )

    alpha_daily = result.intercept

    alpha_annualized = (
        (1 + alpha_daily) ** 252 - 1
    ) * 100 if alpha_daily > -1 else np.nan

    return {
        "Alpha Daily (%)": alpha_daily * 100,
        "Alpha Annualized (%)": alpha_annualized,
        "Beta": result.slope,
        "R Squared": result.rvalue ** 2,
        "P Value": result.pvalue,
        "Standard Error": result.stderr,
        "Observations": len(aligned)
    }


# ==================================================
# 2. AUGMENTED DICKEY-FULLER TEST
# ==================================================

def adf_test(series):
    """
    Perform Augmented Dickey-Fuller test.

    Null Hypothesis:
        The series contains a unit root (non-stationary).

    Alternative Hypothesis:
        The series is stationary.
    """

    series = pd.Series(series).dropna()

    if len(series) < 30:
        return None

    if series.nunique() <= 1:
        return None

    result = adfuller(
        series,
        autolag="AIC"
    )

    return {
        "ADF Statistic": result[0],
        "P Value": result[1],
        "Critical Value (1%)": result[4]["1%"],
        "Critical Value (5%)": result[4]["5%"],
        "Critical Value (10%)": result[4]["10%"],
        "Stationary": result[1] < 0.05,
        "Lags Used": result[2],
        "Observations": result[3]
    }


# ==================================================
# 3. AUTOCORRELATION AND PARTIAL AUTOCORRELATION
# ==================================================

def calculate_acf_pacf(series, lags=30):
    """
    Calculate ACF and PACF values.

    ACF:
        Measures correlation between observations
        and their lagged values.

    PACF:
        Measures direct correlation after controlling
        for intermediate lags.
    """

    series = pd.Series(series).dropna()

    if len(series) < 50:
        return None, None

    max_lags = min(
        lags,
        len(series) // 2 - 1
    )

    if max_lags < 1:
        return None, None

    acf_values = acf(
        series,
        nlags=max_lags,
        fft=True
    )

    pacf_values = pacf(
        series,
        nlags=max_lags,
        method="ywm"
    )

    return acf_values, pacf_values


# ==================================================
# 4. GARCH(1,1) VOLATILITY MODEL
# ==================================================

def fit_garch_model(returns):
    """
    Fit a GARCH(1,1) model to daily stock returns.

    Model:

        Sigma(t)^2 =
        Omega + Alpha * Error(t-1)^2
        + Beta * Sigma(t-1)^2

    Returns:
        Model summary
        Estimated parameters
        Conditional volatility
        Convergence status
    """

    returns = pd.Series(returns).dropna() * 100

    if len(returns) < 100:
        return None

    if returns.std() == 0:
        return None

    try:

        model = arch_model(
            returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False
        )

        fitted = model.fit(
            disp="off",
            show_warning=False
        )

        conditional_volatility = (
            fitted.conditional_volatility
        )

        return {
            "summary": fitted.summary(),

            "parameters": fitted.params,

            "conditional_volatility": (
                conditional_volatility
            ),

            "convergence": fitted.convergence_flag
        }

    except Exception as error:

        print(f"GARCH estimation error: {error}")

        return None
