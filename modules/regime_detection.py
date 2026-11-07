
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def detect_market_regimes(
    market_returns,
    volatility_window=20,
    n_clusters=3
):
    """
    Identify historical market regimes using K-Means.

    Features:
        1. Daily market returns
        2. Rolling annualized volatility

    Returns:
        DataFrame containing returns, volatility,
        cluster IDs and interpreted regime labels.
    """

    data = pd.DataFrame(index=market_returns.index)

    data["Return"] = market_returns

    data["Volatility"] = (
        market_returns
        .rolling(volatility_window)
        .std() * np.sqrt(252) * 100
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(data) < max(60, n_clusters * 10):
        return None

    features = data[["Return", "Volatility"]]

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    data["Cluster"] = model.fit_predict(scaled_features)

    # Calculate average characteristics of each cluster.
    cluster_stats = data.groupby("Cluster").agg(
        Average_Return=("Return", "mean"),
        Average_Volatility=("Volatility", "mean")
    )

    # Rank clusters by average market return.
    ranked_clusters = cluster_stats.sort_values(
        "Average_Return"
    )

    labels = {}

    if n_clusters == 3:
        labels = {
            ranked_clusters.index[0]: "Bear",
            ranked_clusters.index[1]: "Sideways",
            ranked_clusters.index[2]: "Bull"
        }

    else:
        for rank, cluster_id in enumerate(
            ranked_clusters.index
        ):
            labels[cluster_id] = f"Regime {rank + 1}"

    data["Regime"] = data["Cluster"].map(labels)

    return data


def calculate_regime_statistics(regime_data):
    """
    Summarize the historical characteristics of each regime.
    """

    if regime_data is None or regime_data.empty:
        return pd.DataFrame()

    summary = regime_data.groupby("Regime").agg(
        Average_Daily_Return=("Return", "mean"),
        Average_Volatility=("Volatility", "mean"),
        Trading_Days=("Return", "count")
    )

    summary["Average_Annualized_Return"] = (
        (1 + summary["Average_Daily_Return"]) ** 252 - 1
    ) * 100

    summary["Average_Daily_Return"] *= 100

    return summary.round(4)
