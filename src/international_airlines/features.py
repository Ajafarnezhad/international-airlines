"""Data loading and feature engineering.

Centralizing this logic fixes the original prototype's most serious bug:
notebook 3 (anomaly detection) loaded ``processed_city_pairs_eda.csv`` --
the output of notebook 1 -- and immediately referenced
``Normalized_Passengers`` / ``Normalized_Freight`` columns. Those columns
were computed by notebook 2 (forecasting) via ``MinMaxScaler``, but notebook
2 never saved its augmented dataframe anywhere; it only wrote out Prophet's
forecast results. Notebook 3 therefore raised ``KeyError`` the moment it
ran, because the file it actually loaded never had those columns.

:func:`build_features` computes everything in one place so every stage of
the pipeline reads from the same, fully-engineered dataset.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import NUMERIC_COLUMNS, REGION_MAP


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(0)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month_num"].astype(str) + "-01")
    return df


def add_region(df: pd.DataFrame) -> pd.DataFrame:
    """Map each row's `Country` to a broader region, defaulting to "Other".

    The original notebook's mapping already handled this correctly with
    ``.fillna('Other')``; kept as-is here, just centralized.
    """
    df = df.copy()
    df["Region"] = df["Country"].map(REGION_MAP).fillna("Other")
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Passenger_Diff"] = df["Passengers_In"] - df["Passengers_Out"]
    # +1e-5 avoids a division-by-zero producing inf on routes with zero
    # outbound freight (present in this dataset -- see config.NUMERIC_COLUMNS
    # min values).
    df["Freight_Ratio"] = df["Freight_In_(tonnes)"] / (df["Freight_Out_(tonnes)"] + 1e-5)
    return df


def add_normalized_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Add 0-1 normalized passenger/freight/mail columns AND a normalized
    `Passenger_Diff`.

    The original notebook normalized passengers/freight/mail but then fed
    `KMeans` a mix of those normalized columns *and* the raw-scale
    `Passenger_Diff` (which ranges into the tens of thousands). Since KMeans
    is a distance-based algorithm, that one unnormalized column would have
    dominated every cluster assignment. `Passenger_Diff` is normalized here
    too, alongside the rest, so clustering weighs all three signals
    comparably.
    """
    df = df.copy()
    scaler = MinMaxScaler()
    source_cols = ["Passengers_Total", "Freight_Total_(tonnes)", "Mail_Total_(tonnes)", "Passenger_Diff"]
    target_cols = ["Normalized_Passengers", "Normalized_Freight", "Normalized_Mail", "Normalized_Passenger_Diff"]
    df[target_cols] = scaler.fit_transform(df[source_cols])
    return df, scaler


def build_features(path: str) -> pd.DataFrame:
    """Load the raw dataset and apply every engineering step used downstream."""
    df = load_raw(path)
    df = add_region(df)
    df = add_derived_metrics(df)
    df, _ = add_normalized_columns(df)
    return df


__all__ = ["add_derived_metrics", "add_normalized_columns", "add_region", "build_features", "load_raw"]
