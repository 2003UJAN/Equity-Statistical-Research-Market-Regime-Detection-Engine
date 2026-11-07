
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

# ==========================================================
# IMPORT PROJECT MODULES
# ==========================================================

from modules.data_loader import (
    download_stock_data,
    prepare_market_data
)

from modules.statistics_engine import (
    calculate_descriptive_statistics,
    calculate_correlation,
    calculate_covariance,
    calculate_rolling_volatility,
    calculate_rolling_correlation
)

from modules.risk_engine import (
    calculate_risk_metrics
)

from modules.econometrics_engine import (
    capm_regression,
    adf_test,
    calculate_acf_pacf,
    fit_garch_model
)

from modules.regime_detection import (
    detect_market_regimes,
    calculate_regime_statistics
)

from modules.gemini_assistant import (
    explain_quantitative_results
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Equity Statistical Research Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ==========================================================
# 2. CUSTOM CSS - DARK INSTITUTIONAL THEME
# ==========================================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background-color: #0E1117;
    color: #F1F5F9;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* All general text */

.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp div[data-testid="stMarkdownContainer"] {
    color: #E2E8F0;
}

/* Headings */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5 {
    color: #F8FAFC !important;
    font-weight: 650;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #161B24;
    border-right: 1px solid #303947;
}

section[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

/* Metrics */

div[data-testid="stMetric"] {
    background-color: #1C2430;
    border: 1px solid #334155;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
}

div[data-testid="stMetricDelta"] {
    color: #38BDF8 !important;
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #0E1117;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1C2430;
    border-radius: 8px;
    padding: 12px 16px;
    color: #CBD5E1 !important;
}

.stTabs [aria-selected="true"] {
    background-color: #26364A !important;
    border-bottom: 2px solid #38BDF8 !important;
}

.stTabs [data-baseweb="tab"] p {
    color: #E2E8F0 !important;
}

/* Input fields */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: #1C2430 !important;
    color: #F8FAFC !important;
    border: 1px solid #475569 !important;
}

/* Select boxes */

div[data-baseweb="select"] > div {
    background-color: #1C2430 !important;
    color: #F8FAFC !important;
    border-color: #475569 !important;
}

/* Dropdown text */

div[data-baseweb="popover"] * {
    color: #F1F5F9 !important;
}

/* Buttons */

.stButton button {
    background-color: #2563EB;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 16px;
}

.stButton button:hover {
    background-color: #1D4ED8;
    border: 1px solid #60A5FA;
    color: #FFFFFF !important;
}

/* Download buttons */

.stDownloadButton button {
    background-color: #1C2430;
    color: #F8FAFC !important;
    border: 1px solid #475569;
    border-radius: 8px;
}

/* Dataframes */

div[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
}

/* Expanders */

.streamlit-expanderHeader {
    background-color: #1C2430;
    color: #F8FAFC !important;
}

/* Alerts */

div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Horizontal dividers */

hr {
    border-color: #334155 !important;
}

/* Captions */

.stCaption,
div[data-testid="stCaptionContainer"] {
    color: #94A3B8 !important;
}

/* Chat messages */

div[data-testid="stChatMessage"] {
    background-color: #1C2430;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px;
}

/* Footer */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. APPLICATION HEADER
# ==========================================================

st.title("📈 Equity Statistical Research Engine")

st.markdown("""
An interactive quantitative equity research platform
for statistical market analysis, financial econometrics,
risk measurement, volatility modelling and machine learning
based market regime detection.
""")

st.caption(
    "QUANTITATIVE RESEARCH  |  FINANCIAL ECONOMETRICS  |  "
    "RISK ANALYTICS  |  MACHINE LEARNING  |  GENERATIVE AI"
)

st.divider()


# ==========================================================
# 4. SIDEBAR CONFIGURATION
# ==========================================================

st.sidebar.title("Research Configuration")

st.sidebar.markdown("### Stock Universe")

stock_options = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "AXISBANK.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "WIPRO.NS",
    "TATAMOTORS.NS",
    "ASIANPAINT.NS"
]

default_stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS"
]

selected_stocks = st.sidebar.multiselect(
    "Select stocks",
    options=stock_options,
    default=default_stocks
)

benchmark = st.sidebar.selectbox(
    "Select benchmark",
    options=["^NSEI", "^NSEBANK"],
    format_func=lambda x: {
        "^NSEI": "Nifty 50",
        "^NSEBANK": "Nifty Bank"
    }[x]
)

period = st.sidebar.selectbox(
    "Historical period",
    ["6mo", "1y", "2y", "5y", "10y"],
    index=2
)

risk_free_rate = st.sidebar.number_input(
    "Annual risk-free rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=6.0,
    step=0.25
) / 100

volatility_window = st.sidebar.selectbox(
    "Rolling volatility window",
    [20, 30, 60, 90],
    index=0
)

st.sidebar.divider()

st.sidebar.caption(
    "Data Source: Yahoo Finance\n\n"
    "Frequency: Daily\n\n"
    "Annualization: 252 Trading Days"
)


# ==========================================================
# 5. DATA DOWNLOAD
# ==========================================================

if not selected_stocks:

    st.warning("Please select at least one stock.")

    st.stop()


tickers = list(dict.fromkeys(
    selected_stocks + [benchmark]
))

with st.spinner("Downloading historical market data..."):

    try:

        raw_prices = download_stock_data(
            tickers,
            period=period,
            interval="1d"
        )

    except Exception as error:

        st.error(f"Data download failed: {error}")

        st.stop()


if raw_prices.empty:

    st.error(
        "Unable to retrieve financial market data. "
        "Please check your internet connection or try again."
    )

    st.stop()


if benchmark not in raw_prices.columns:

    st.error(
        "Benchmark data is unavailable. "
        "Please select another benchmark."
    )

    st.stop()


# Identify available stocks

available_stocks = [
    stock for stock in selected_stocks
    if stock in raw_prices.columns
    and raw_prices[stock].notna().sum() >= 30
]


if not available_stocks:

    st.error("No selected stocks have sufficient historical data.")

    st.stop()


# ==========================================================
# 6. DATA PREPROCESSING
# ==========================================================

prices, returns = prepare_market_data(
    raw_prices[[*available_stocks, benchmark]]
)

benchmark_returns = returns[benchmark].dropna()

benchmark_prices = prices[benchmark].dropna()


if len(benchmark_returns) < 30:

    st.error(
        "Insufficient benchmark observations "
        "for quantitative analysis."
    )

    st.stop()


# ==========================================================
# 7. CENTRALIZED QUANTITATIVE CALCULATIONS
# ==========================================================

with st.spinner("Calculating quantitative research metrics..."):

    # Descriptive statistics

    descriptive_stats = calculate_descriptive_statistics(
        returns[available_stocks]
    )

    # Correlation

    correlation = calculate_correlation(
        returns[[*available_stocks, benchmark]]
    )

    # Covariance

    covariance = calculate_covariance(
        returns[[*available_stocks, benchmark]]
    )

    # Rolling volatility

    rolling_volatility = calculate_rolling_volatility(
        returns[available_stocks],
        window=volatility_window
    )

    # Risk metrics

    risk_metrics = calculate_risk_metrics(
        prices[available_stocks],
        returns[available_stocks],
        benchmark_returns,
        risk_free_rate
    )

    # Market regimes

    regime_data = detect_market_regimes(
        benchmark_returns,
        volatility_window=volatility_window,
        n_clusters=3
    )

    if regime_data is not None:

        regime_summary = calculate_regime_statistics(
            regime_data
        )

    else:

        regime_summary = pd.DataFrame()


# ==========================================================
# 8. BENCHMARK PERFORMANCE
# ==========================================================

elapsed_days = (
    benchmark_prices.index[-1] -
    benchmark_prices.index[0]
).days

if len(benchmark_prices) > 1 and elapsed_days > 0:

    benchmark_cagr = (
        (
            benchmark_prices.iloc[-1] /
            benchmark_prices.iloc[0]
        ) ** (365.25 / elapsed_days) - 1
    ) * 100

else:

    benchmark_cagr = np.nan


benchmark_volatility = (
    benchmark_returns.std() * np.sqrt(252) * 100
)

daily_rf = (
    (1 + risk_free_rate) ** (1 / 252) - 1
)

benchmark_sharpe = (

    (benchmark_returns.mean() - daily_rf)
    / benchmark_returns.std()
    * np.sqrt(252)

    if benchmark_returns.std() > 0
    else np.nan
)


# ==========================================================
# 9. MAIN DASHBOARD NAVIGATION
# ==========================================================

tabs = st.tabs([
    "Market Overview",
    "Statistical Analysis",
    "Correlation Engine",
    "Risk Analytics",
    "CAPM & Regression",
    "Time Series",
    "Regime Detection",
    "Gemini AI Assistant"
])


# ==========================================================
# TAB 1: MARKET OVERVIEW
# ==========================================================

with tabs[0]:

    st.subheader("Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    benchmark_name = (
        "Nifty 50"
        if benchmark == "^NSEI"
        else "Nifty Bank"
    )

    col1.metric(
        "Benchmark",
        benchmark_name
    )

    col2.metric(
        "Latest Index Price",
        f"{benchmark_prices.iloc[-1]:,.2f}"
    )

    col3.metric(
        "Realized CAGR",
        f"{benchmark_cagr:.2f}%"
        if pd.notna(benchmark_cagr)
        else "N/A"
    )

    col4.metric(
        "Annualized Volatility",
        f"{benchmark_volatility:.2f}%"
    )

    st.divider()

    # Normalized price chart

    st.subheader("Normalized Price Performance")

    first_prices = prices.apply(
        lambda x: x.dropna().iloc[0]
        if not x.dropna().empty
        else np.nan
    )

    normalized_prices = prices.divide(
        first_prices,
        axis=1
    ) * 100

    fig = px.line(
        normalized_prices,
        title="Historical Stock Performance (Base = 100)",
        labels={
            "value": "Normalized Price",
            "index": "Date",
            "variable": "Stock"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Daily returns

    st.subheader("Daily Stock Returns")

    fig = px.line(
        returns[available_stocks],
        title="Daily Returns",
        labels={
            "value": "Daily Return",
            "index": "Date",
            "variable": "Stock"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Latest prices

    st.subheader("Latest Stock Prices")

    st.dataframe(
        prices[available_stocks].tail(10),
        use_container_width=True
    )

    st.metric(
        "Benchmark Sharpe Ratio",
        f"{benchmark_sharpe:.3f}"
        if pd.notna(benchmark_sharpe)
        else "N/A"
    )


# ==========================================================
# TAB 2: STATISTICAL ANALYSIS
# ==========================================================

with tabs[1]:

    st.subheader("Descriptive Statistical Analysis")

    st.markdown(
        "Statistical characteristics of historical daily returns."
    )

    st.dataframe(
        descriptive_stats,
        use_container_width=True
    )

    st.download_button(
        "Download Statistical Analysis",
        descriptive_stats.to_csv().encode("utf-8"),
        "statistical_analysis.csv",
        "text/csv"
    )

    st.divider()

    # Return distribution

    st.subheader("Return Distribution")

    selected_ticker = st.selectbox(
        "Select stock",
        available_stocks,
        key="distribution_stock"
    )

    fig = px.histogram(
        returns[selected_ticker].dropna(),
        nbins=50,
        marginal="box",
        title=f"Daily Return Distribution: {selected_ticker}"
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Daily Return",
        yaxis_title="Frequency"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Rolling volatility

    st.divider()

    st.subheader("Rolling Volatility")

    fig = px.line(
        rolling_volatility,
        title=f"{volatility_window}-Day Annualized Volatility"
    )

    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Volatility (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# TAB 3: CORRELATION ENGINE
# ==========================================================

with tabs[2]:

    st.subheader("Correlation Matrix")

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Heatmap"
    )

    fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "Download Correlation Matrix",
        correlation.to_csv().encode("utf-8"),
        "correlation_matrix.csv",
        "text/csv"
    )

    st.divider()

    st.subheader("Covariance Matrix")

    fig = px.imshow(
        covariance,
        text_auto=".5f",
        color_continuous_scale="Viridis",
        title="Daily Return Covariance Matrix"
    )

    fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "Download Covariance Matrix",
        covariance.to_csv().encode("utf-8"),
        "covariance_matrix.csv",
        "text/csv"
    )

    st.divider()

    # Rolling correlation

    st.subheader("Rolling Correlation")

    selected_stock = st.selectbox(
        "Select stock",
        available_stocks,
        key="rolling_corr_stock"
    )

    rolling_corr = calculate_rolling_correlation(
        returns,
        selected_stock,
        benchmark,
        window=60
    )

    fig = px.line(
        rolling_corr,
        title=f"60-Day Rolling Correlation: {selected_stock}"
    )

    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Correlation",
        yaxis_range=[-1, 1]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# TAB 4: RISK ANALYTICS
# ==========================================================

with tabs[3]:

    st.subheader("Quantitative Risk Analytics")

    st.markdown(
        "Evaluate historical risk and risk-adjusted performance."
    )

    st.dataframe(
        risk_metrics,
        use_container_width=True
    )

    st.download_button(
        "Download Risk Metrics",
        risk_metrics.to_csv().encode("utf-8"),
        "risk_metrics.csv",
        "text/csv"
    )

    st.divider()

    # Beta analysis

    st.subheader("Beta Analysis")

    fig = px.bar(
        risk_metrics.reset_index(),
        x="index",
        y="Beta",
        title=f"Stock Beta Relative to {benchmark_name}"
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Stock",
        yaxis_title="Beta"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # Risk-return analysis

    st.subheader("Risk vs Return")

    risk_plot_data = risk_metrics.reset_index()

    risk_plot_data["CVaR Size"] = (
        risk_plot_data["95% CVaR (%)"]
        .fillna(0)
        .clip(lower=0.01)
    )

    fig = px.scatter(
        risk_plot_data,
        x="Annualized Volatility (%)",
        y="Realized CAGR (%)",
        color="Sharpe Ratio",
        size="CVaR Size",
        hover_name="index",
        title="Risk-Return Profile"
    )

    fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# TAB 5: CAPM AND REGRESSION
# ==========================================================

with tabs[4]:

    st.subheader("Capital Asset Pricing Model")

    st.markdown(
        "Estimate stock sensitivity and excess-return "
        "relationship relative to the selected benchmark."
    )

    capm_stock = st.selectbox(
        "Select stock",
        available_stocks,
        key="capm_stock"
    )

    capm_result = capm_regression(
        returns[capm_stock],
        benchmark_returns,
        risk_free_rate
    )

    if capm_result is not None:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Beta",
            f"{capm_result['Beta']:.4f}"
        )

        c2.metric(
            "R Squared",
            f"{capm_result['R Squared']:.4f}"
        )

        c3.metric(
            "P-Value",
            f"{capm_result['P Value']:.6f}"
        )

        st.divider()

        st.subheader("Regression Results")

        st.dataframe(
            pd.DataFrame(
                capm_result.items(),
                columns=["Metric", "Value"]
            ),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Regression scatter plot

        aligned = pd.concat(
            [
                returns[capm_stock],
                benchmark_returns
            ],
            axis=1
        ).dropna()

        aligned.columns = [
            "Stock Returns",
            "Benchmark Returns"
        ]

        fig = px.scatter(
            aligned,
            x="Benchmark Returns",
            y="Stock Returns",
            trendline="ols",
            title=f"Stock vs Benchmark Regression: {capm_stock}"
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "CAPM uses daily excess returns and a constant "
            "annual risk-free rate converted to a daily rate. "
            "Alpha is annualized from the daily intercept."
        )

    else:

        st.warning(
            "Insufficient observations for CAPM regression."
        )


# ==========================================================
# TAB 6: TIME SERIES ANALYSIS
# ==========================================================

with tabs[5]:

    st.subheader("Time Series Econometrics")

    time_series_stock = st.selectbox(
        "Select stock",
        available_stocks,
        key="timeseries_stock"
    )

    series_type = st.radio(
        "Select series",
        [
            "Daily Returns",
            "Log Returns",
            "Closing Prices"
        ],
        horizontal=True
    )

    if series_type == "Daily Returns":

        series = returns[time_series_stock].dropna()

    elif series_type == "Log Returns":

        series = np.log(
            prices[time_series_stock] /
            prices[time_series_stock].shift(1)
        ).dropna()

    else:

        series = prices[time_series_stock].dropna()

    st.divider()

    # ADF test

    st.subheader("Augmented Dickey-Fuller Test")

    adf_result = adf_test(series)

    if adf_result is not None:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "ADF Statistic",
            f"{adf_result['ADF Statistic']:.4f}"
        )

        c2.metric(
            "P-Value",
            f"{adf_result['P Value']:.6f}"
        )

        c3.metric(
            "Stationarity",
            "Yes"
            if adf_result["Stationary"]
            else "No"
        )

        st.dataframe(
            pd.DataFrame(
                adf_result.items(),
                columns=["Metric", "Value"]
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Insufficient or constant observations for ADF testing."
        )

    st.divider()

    # ACF and PACF

    st.subheader("Autocorrelation and Partial Autocorrelation")

    acf_values, pacf_values = calculate_acf_pacf(
        series,
        lags=30
    )

    if acf_values is not None and pacf_values is not None:

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 4)
        )

        axes[0].stem(
            range(len(acf_values)),
            acf_values
        )

        axes[0].set_title("Autocorrelation Function")

        axes[0].set_xlabel("Lag")

        axes[0].set_ylabel("ACF")

        axes[1].stem(
            range(len(pacf_values)),
            pacf_values
        )

        axes[1].set_title("Partial Autocorrelation Function")

        axes[1].set_xlabel("Lag")

        axes[1].set_ylabel("PACF")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.warning(
            "Insufficient observations for ACF/PACF analysis."
        )

    st.divider()

    # GARCH model

    st.subheader("GARCH(1,1) Volatility Model")

    garch_result = fit_garch_model(
        returns[time_series_stock]
    )

    if garch_result is not None:

        st.markdown("**Estimated Model Parameters**")

        st.dataframe(
            garch_result["parameters"].to_frame(
                name="Estimated Value"
            ),
            use_container_width=True
        )

        if garch_result["convergence"] == 0:

            st.success("GARCH model converged successfully.")

        else:

            st.warning(
                "GARCH optimizer did not report successful convergence."
            )

        fig = px.line(
            garch_result["conditional_volatility"],
            title="GARCH Conditional Volatility"
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis_title="Daily Volatility (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        with st.expander("View GARCH Model Summary"):

            st.text(
                str(garch_result["summary"])
            )

    else:

        st.warning(
            "GARCH estimation failed or insufficient observations."
        )


# ==========================================================
# TAB 7: MARKET REGIME DETECTION
# ==========================================================

with tabs[6]:

    st.subheader("Machine Learning-Based Market Regime Detection")

    st.markdown("""
    K-Means clustering identifies historical market regimes
    using daily market returns and rolling annualized volatility.

    The clusters are interpreted retrospectively as Bull,
    Bear and Sideways regimes based on their average returns.
    """)

    if regime_data is not None:

        st.subheader("Historical Regime Statistics")

        st.dataframe(
            regime_summary,
            use_container_width=True
        )

        st.divider()

        regime_plot = (
            regime_data
            .rename_axis("Date")
            .reset_index()
        )

        # Regime timeline

        st.subheader("Market Regime Timeline")

        fig = px.scatter(
            regime_plot,
            x="Date",
            y="Return",
            color="Regime",
            hover_data=[
                "Volatility",
                "Cluster"
            ],
            title="Historical Market Regimes"
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis_title="Daily Market Return"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # Regime distribution

        st.subheader("Regime Distribution")

        regime_counts = (
            regime_data["Regime"]
            .value_counts()
            .rename_axis("Regime")
            .reset_index(name="Trading Days")
        )

        fig = px.pie(
            regime_counts,
            names="Regime",
            values="Trading Days",
            hole=0.4,
            title="Historical Regime Distribution"
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # Return versus volatility

        st.subheader("Market Return vs Volatility")

        fig = px.scatter(
            regime_plot,
            x="Volatility",
            y="Return",
            color="Regime",
            hover_data=["Date"],
            title="Market Regime Characteristics"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Annualized Volatility (%)",
            yaxis_title="Daily Return"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.download_button(
            "Download Regime Analysis",
            regime_data.to_csv().encode("utf-8"),
            "market_regime_analysis.csv",
            "text/csv"
        )

        st.caption(
            "Regime classifications are sample-dependent "
            "and are not forward-looking predictions."
        )

    else:

        st.warning(
            "Insufficient observations for market regime detection."
        )


# ==========================================================
# TAB 8: GEMINI AI QUANT RESEARCH ASSISTANT
# ==========================================================

with tabs[7]:

    st.subheader("🤖 Gemini AI Quant Research Assistant")

    st.markdown("""
    An AI-powered research assistant that interprets
    quantitative financial results using Google Gemini.

    Python performs the calculations. Gemini explains
    the results in professional financial research language.
    """)

    st.divider()

    # ------------------------------------------------------
    # STOCK SELECTION
    # ------------------------------------------------------

    ai_stock = st.selectbox(
        "Select stock for AI research",
        available_stocks,
        key="gemini_stock"
    )

    # ------------------------------------------------------
    # PREPARE STOCK-SPECIFIC RESEARCH CONTEXT
    # ------------------------------------------------------

    ai_capm = capm_regression(
        returns[ai_stock],
        benchmark_returns,
        risk_free_rate
    )

    ai_context = {

        "Stock": ai_stock,

        "Benchmark": benchmark_name,

        "Historical Period": period,

        "Risk Free Rate": risk_free_rate,

        "Descriptive Statistics": (
            descriptive_stats.loc[ai_stock].to_dict()
        ),

        "Risk Metrics": (
            risk_metrics.loc[ai_stock].to_dict()
        ),

        "CAPM Results": (
            ai_capm if ai_capm is not None else {}
        ),

        "Market Regime Analysis": (
            regime_summary.to_dict()
            if not regime_summary.empty
            else {}
        )
    }

    # ------------------------------------------------------
    # RESEARCH QUESTION SELECTION
    # ------------------------------------------------------

    st.subheader("Research Questions")

    suggestions = [

        "Explain the overall risk profile of this stock.",

        "Interpret its beta, alpha and CAPM regression.",

        "Explain its volatility and Sharpe Ratio.",

        "What do skewness and kurtosis indicate?",

        "Explain the market regime analysis.",

        "Generate a complete quantitative equity research note."

    ]

    selected_question = st.selectbox(
        "Choose a suggested question",
        ["Custom Question"] + suggestions,
        key="gemini_suggestion"
    )

    custom_question = st.text_area(
        "Enter your research question",
        placeholder=(
            "Example: Why does this stock have a high beta? "
            "Explain its risk characteristics and market sensitivity."
        ),
        height=100,
        key="gemini_custom_question"
    )

    # ------------------------------------------------------
    # CONVERSATION HISTORY
    # ------------------------------------------------------

    history_key = f"gemini_history_{ai_stock}"

    if history_key not in st.session_state:

        st.session_state[history_key] = []

    history = st.session_state[history_key]

    # Display previous messages

    for message in history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # ------------------------------------------------------
    # ACTION BUTTONS
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        generate = st.button(
            "Generate AI Explanation",
            type="primary",
            use_container_width=True
        )

    with col2:

        clear = st.button(
            "Clear Conversation",
            use_container_width=True
        )

    # ------------------------------------------------------
    # CLEAR CHAT
    # ------------------------------------------------------

    if clear:

        st.session_state[history_key] = []

        st.rerun()

    # ------------------------------------------------------
    # GEMINI API CALL
    # ------------------------------------------------------

    if generate:

        if selected_question == "Custom Question":

            question = custom_question.strip()

        else:

            question = selected_question

        if not question:

            st.warning(
                "Please enter a research question."
            )

        else:

            with st.spinner(
                "Gemini is analysing quantitative results..."
            ):

                answer = explain_quantitative_results(
                    research_context=ai_context,
                    question=question,
                    conversation_history=history
                )

            # Save conversation

            history.append({
                "role": "user",
                "content": question
            })

            history.append({
                "role": "assistant",
                "content": answer
            })

            st.session_state[history_key] = history[-12:]

            st.rerun()

    # ------------------------------------------------------
    # DOWNLOAD RESEARCH CONTEXT
    # ------------------------------------------------------

    st.divider()

    st.subheader("Research Data")

    with st.expander("View quantitative data sent to Gemini"):

        st.json(ai_context)

    st.download_button(
        "Download AI Research Context",
        pd.Series(ai_context).to_json(
            indent=4,
            default_handler=str
        ),
        f"{ai_stock}_research_context.json",
        "application/json"
    )

    st.caption(
        "Powered by Google Gemini. "
        "All quantitative calculations are performed independently "
        "using Python. AI-generated explanations are for research "
        "and educational purposes only."
    )


# ==========================================================
# 10. FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#8993A2;">

    <h4>Equity Statistical Research & Market Regime Detection Engine</h4>

    Built using Python | Streamlit | NumPy | Pandas |
    Scikit-learn | Statsmodels | ARCH | Plotly | Google Gemini

    <br><br>

    Quantitative Research | Financial Econometrics |
    Risk Analytics | Machine Learning | Generative AI

    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Historical market data: Yahoo Finance. "
    "Historical analysis does not guarantee future performance. "
    "This application is intended for quantitative research "
    "and educational purposes, not investment advice."
)
