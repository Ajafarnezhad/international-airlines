from __future__ import annotations

from pathlib import Path

from international_airlines.anomaly import cluster_ports, detect_anomalies
from international_airlines.features import build_features


def test_detect_anomalies_flags_a_minority(raw_csv: Path):
    df = build_features(str(raw_csv))
    result = detect_anomalies(df, contamination=0.1)
    assert set(result["Anomaly"].unique()) <= {1, -1}
    assert (result["Anomaly"] == -1).sum() < len(result)


def test_cluster_ports_uses_normalized_features_not_raw_diff(raw_csv: Path):
    df = build_features(str(raw_csv))
    clusters = cluster_ports(df, n_clusters=2)

    assert set(clusters["ForeignPort"]) == set(df["ForeignPort"].unique())
    assert clusters["Cluster"].nunique() <= 2

    # Regression guard: clustering must run on the *normalized* Passenger_Diff
    # column, which is bounded to [0, 1] -- unlike the raw column, which can
    # be in the thousands and would dominate KMeans distances if used
    # directly (the original notebook's bug).
    from international_airlines.anomaly import CLUSTER_FEATURES

    assert "Normalized_Passenger_Diff" in CLUSTER_FEATURES
    assert "Passenger_Diff" not in CLUSTER_FEATURES
