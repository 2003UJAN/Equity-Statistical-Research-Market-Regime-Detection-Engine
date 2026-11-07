# 📈 Equity Statistical Research & Market Regime Detection Engine

**An interactive quantitative equity research platform built using Python, Streamlit, Financial Econometrics, Statistical Modelling, Machine Learning and Generative AI.**

The platform provides an integrated environment for analysing equity market behaviour, measuring financial risk, conducting econometric analysis, identifying market regimes and generating AI-powered quantitative research insights.

---

## 🚀 Project Overview

The **Equity Statistical Research & Market Regime Detection Engine** is designed to replicate key analytical workflows used in quantitative equity research and institutional financial markets.

It combines historical market data, statistical techniques, financial econometrics and machine learning to investigate stock performance, market relationships, volatility dynamics and risk characteristics.

The application also integrates **Google Gemini AI** to interpret Python-calculated quantitative outputs and generate structured financial research explanations.

## 🎯 Key Features

### 1. Market Overview
- Historical stock price analysis.
- Normalized stock performance comparison (Base = 100).
- Daily stock return visualization.
- Benchmark performance tracking.
- Annualized volatility and CAGR analysis.
- Interactive financial charts using Plotly.

### 2. Statistical Analysis
- Descriptive statistical analysis of equity returns.
- Mean, median, standard deviation, skewness and kurtosis.
- Return distribution and histogram analysis.
- Rolling volatility estimation.
- Historical statistical comparison across multiple equities.

### 3. Correlation & Covariance Engine
- Pearson correlation matrix.
- Covariance matrix estimation.
- Interactive correlation heatmaps.
- Rolling correlation analysis.
- Identification of changing relationships between individual equities and market benchmarks.

### 4. Quantitative Risk Analytics
- Beta estimation.
- Annualized volatility.
- Sharpe Ratio.
- Historical Value at Risk (VaR).
- Conditional Value at Risk (CVaR).
- Risk-adjusted performance analysis.
- Risk-return visualization.

### 5. CAPM & Regression Analysis
- Capital Asset Pricing Model (CAPM).
- Market beta estimation.
- Daily and annualized alpha estimation.
- R-squared and regression significance.
- Stock versus benchmark regression analysis.
- Excess-return-based regression modelling.

### 6. Time Series & Financial Econometrics
- Augmented Dickey-Fuller (ADF) stationarity testing.
- Autocorrelation Function (ACF).
- Partial Autocorrelation Function (PACF).
- GARCH(1,1) conditional volatility modelling.
- Time-series analysis of stock prices and returns.
- Volatility dynamics and clustering analysis.

### 7. Machine Learning-Based Market Regime Detection
- K-Means clustering for market regime identification.
- Classification of historical market conditions into Bull, Bear and Sideways regimes based on cluster characteristics.
- Return and volatility-based regime analysis.
- Historical regime distribution.
- Market regime timeline visualization.
- Comparative analysis of market behaviour across different regimes.

### 8. Gemini AI Quantitative Research Assistant
- Google Gemini API integration.
- Natural language interpretation of quantitative financial metrics.
- CAPM, alpha and beta explanations.
- Statistical distribution interpretation.
- Volatility and risk analysis.
- Market regime interpretation.
- AI-generated quantitative equity research notes.
- Conversational research interface with stock-specific context.

**Note:** Python performs all financial and statistical calculations independently. Gemini is used exclusively for interpreting the calculated outputs and generating research-oriented explanations.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend & Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Financial Data | Yahoo Finance (yfinance) |
| Statistical Analysis | SciPy |
| Econometrics | Statsmodels |
| Volatility Modelling | ARCH |
| Machine Learning | Scikit-learn |
| Data Visualization | Plotly, Matplotlib |
| Generative AI | Google Gemini API |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

---

## 📂 Project Structure

```text
Equity-Statistical-Research/
│
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
│
├── assets/
│   └── .gitkeep
│
├── modules/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── econometrics_engine.py
│   ├── gemini_assistant.py
│   ├── regime_detection.py
│   ├── risk_engine.py
│   └── statistics_engine.py
│
├── .gitignore
├── app.py
├── requirements.txt
└── README.md
```

### Module Descriptions

| Module | Description |
|---|---|
| `app.py` | Main Streamlit application and dashboard interface |
| `data_loader.py` | Historical stock data retrieval and preprocessing |
| `statistics_engine.py` | Statistical calculations, correlation and rolling analysis |
| `risk_engine.py` | Financial risk and risk-adjusted performance metrics |
| `econometrics_engine.py` | CAPM, ADF testing, ACF/PACF and GARCH modelling |
| `regime_detection.py` | Machine learning-based market regime identification |
| `gemini_assistant.py` | Gemini-powered quantitative research explanations |

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or above (use a version compatible with the installed dependencies).
- Visual Studio Code.
- Git.
- Internet connection for retrieving historical market data.
- Google Gemini API key (optional, for AI-powered explanations).

### Step 1: Clone the Repository

```bash
git clone https://github.com/2003UJAN/Equity-Statistical-Research-Market-Regime-Detection-Engine.git
```

Navigate to the project directory:

```bash
cd Equity-Statistical-Research-Market-Regime-Detection-Engine
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### Step 4: Configure Gemini API

Create a file named `secrets.toml` inside the `.streamlit` directory.

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-flash-latest"
```

Obtain an API key through [Google AI Studio](https://aistudio.google.com/apikey).

Ensure that the API project has the required access and billing configuration.

**Security:** Never upload `secrets.toml` or expose your API key publicly.

### Step 5: Run the Application

```bash
python -m streamlit run app.py
```

The application will be accessible at:

```text
http://localhost:8501
```

---

## 📊 Supported Market Instruments

The current application includes selected Indian equities:

- Reliance Industries
- Tata Consultancy Services
- Infosys
- HDFC Bank
- ICICI Bank
- ITC
- Larsen & Toubro
- State Bank of India
- Bharti Airtel
- Axis Bank
- Maruti Suzuki
- Sun Pharmaceutical
- Wipro
- Tata Motors
- Asian Paints

**Market Benchmarks:**
- Nifty 50 (^NSEI)
- Nifty Bank (^NSEBANK)

Historical data can be analysed across multiple time horizons, including 6 months, 1 year, 2 years, 5 years and 10 years, subject to data availability.

---

## 📐 Quantitative Methodologies

The platform incorporates the following quantitative finance concepts:

| Methodology | Application |
|---|---|
| Descriptive Statistics | Return distribution and statistical characteristics |
| Pearson Correlation | Equity relationship analysis |
| Covariance | Joint return variability |
| CAPM | Systematic risk and excess return analysis |
| Sharpe Ratio | Risk-adjusted performance |
| VaR & CVaR | Downside risk measurement |
| ADF Test | Time-series stationarity |
| ACF & PACF | Return autocorrelation analysis |
| GARCH(1,1) | Conditional volatility modelling |
| K-Means Clustering | Market regime identification |

---

## 🔐 Configuration & Security

- API credentials are maintained through Streamlit Secrets.
- Sensitive configuration files are excluded through `.gitignore`.
- Python-based calculations remain independent of the Generative AI layer.
- Historical data availability depends on the external data provider.
- AI-generated explanations should be independently verified.

---

## ⚠️ Limitations

- Historical market data is sourced from Yahoo Finance and may contain missing observations or data delays.
- Statistical estimates depend on the selected historical period and data quality.
- Market regime classifications are retrospective and sample-dependent.
- GARCH estimates depend on model assumptions and convergence.
- CAPM relies on a constant risk-free rate in the current implementation.
- Gemini API availability depends on API permissions, billing and usage quotas.
- Historical performance does not guarantee future returns.

---

## 🔮 Future Enhancements

- Fama-French Three-Factor and Five-Factor Models.
- Momentum, Value, Quality and Low-Volatility factor analysis.
- Multi-factor equity screening.
- Portfolio optimization using Modern Portfolio Theory.
- Efficient Frontier construction.
- Monte Carlo portfolio simulations.
- Walk-forward validation and out-of-sample backtesting.
- Advanced volatility forecasting.
- Automated quantitative research report generation.
- Integration of additional financial and macroeconomic datasets.

---

## 👨‍💻 Author

**Ujan Pradhan**

B.Tech – Computer Science & Engineering

Interested in Quantitative Research, Institutional Equities, Financial Econometrics, Algorithmic Trading, Risk Analytics and Applied Machine Learning.

**GitHub:** [2003UJAN](https://github.com/2003UJAN)

---

## 📌 Disclaimer

This project is intended for educational, analytical and quantitative research purposes only. It does not constitute investment advice, a recommendation to buy or sell securities, or a guarantee of future market performance.

Market analysis and model outputs are based on historical data and underlying statistical assumptions.