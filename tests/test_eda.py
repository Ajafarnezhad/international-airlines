from __future__ import annotations

from pathlib import Path

from international_airlines.eda import (
    plot_region_distribution,
    plot_top_countries,
    plot_yearly_trends,
    region_passenger_totals,
    top_countries_by_passengers,
    yearly_trends,
)
from international_airlines.features import build_features


def test_top_countries_by_passengers(raw_csv: Path):
    df = build_features(str(raw_csv))
    top = top_countries_by_passengers(df, top_n=1)
    assert len(top) == 1


def test_region_passenger_totals_sums_to_total(raw_csv: Path):
    df = build_features(str(raw_csv))
    totals = region_passenger_totals(df)
    assert totals["Passengers_Total"].sum() == df["Passengers_Total"].sum()


def test_yearly_trends_covers_all_years(raw_csv: Path):
    df = build_features(str(raw_csv))
    trends = yearly_trends(df)
    assert set(trends["Year"]) == set(df["Year"])


def test_plot_functions_write_files(raw_csv: Path, tmp_path: Path):
    df = build_features(str(raw_csv))
    assert plot_top_countries(df, tmp_path).exists()
    assert plot_region_distribution(df, tmp_path).exists()
    assert plot_yearly_trends(df, tmp_path).exists()
