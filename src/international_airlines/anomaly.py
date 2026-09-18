"""Anomaly detection (IsolationForest) and port clustering (KMeans)."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from .config import RANDOM_STATE

ANOMALY_FEATURES = ["Normalized_Passengers", "Normalized_Freight", "Normalized_Passenger_Diff"]
CLUSTER_FEATURES = ["Normalized_Passengers", "Normalized_Freight", "Normalized_Passenger_Diff"]


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    df["Anomaly"] = model.fit_predict(df[ANOMALY_FEATURES])
    return df


def cluster_ports(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Cluster foreign ports by average traffic behavior.

    Uses the *normalized* `Passenger_Diff` alongside the normalized
    passenger/freight columns (see `features.add_normalized_columns`) --
    the original notebook mixed normalized columns with the raw-scale
    `Passenger_Diff`, which would have dominated every KMeans distance
    calculation.
    """
    port_features = df.groupby("ForeignPort", as_index=False)[CLUSTER_FEATURES].mean()
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    port_features["Cluster"] = kmeans.fit_predict(port_features[CLUSTER_FEATURES])
    return port_features


__all__ = ["ANOMALY_FEATURES", "CLUSTER_FEATURES", "cluster_ports", "detect_anomalies"]
