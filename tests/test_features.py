from __future__ import annotations

from pathlib import Path

from international_airlines.features import add_normalized_columns, build_features, load_raw


def test_load_raw_fills_missing_and_builds_date(raw_csv: Path):
    df = load_raw(str(raw_csv))
    assert "Date" in df.columns
    assert df["Date"].dt.day.eq(1).all()
    assert not df[["Passengers_In", "Freight_In_(tonnes)"]].isna().any().any()


def test_build_features_maps_known_and_unknown_countries(raw_csv: Path):
    df = build_features(str(raw_csv))
    assert set(df.loc[df["Country"] == "New Zealand", "Region"]) == {"Oceania"}
    # "Nowhereland" isn't in REGION_MAP -- must fall back to "Other", not NaN/crash.
    assert set(df.loc[df["Country"] == "Nowhereland", "Region"]) == {"Other"}


def test_build_features_computes_derived_metrics(raw_csv: Path):
    df = build_features(str(raw_csv))
    assert (df["Passenger_Diff"] == df["Passengers_In"] - df["Passengers_Out"]).all()
    assert df["Freight_Ratio"].notna().all()
    assert (df["Freight_Ratio"] >= 0).all()


def test_normalized_columns_are_in_unit_range(raw_csv: Path):
    df = build_features(str(raw_csv))
    for col in ["Normalized_Passengers", "Normalized_Freight", "Normalized_Mail", "Normalized_Passenger_Diff"]:
        assert df[col].min() >= 0.0
        assert df[col].max() <= 1.0


def test_add_normalized_columns_returns_fitted_scaler(raw_csv: Path):
    df = load_raw(str(raw_csv))
    df["Passenger_Diff"] = df["Passengers_In"] - df["Passengers_Out"]
    _, scaler = add_normalized_columns(df)
    assert scaler.data_min_.shape == (4,)
