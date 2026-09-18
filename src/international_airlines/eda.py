"""Exploratory data analysis: aggregations and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def top_countries_by_passengers(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    agg = df.groupby("Country", as_index=False)["Passengers_Total"].sum()
    return agg.sort_values("Passengers_Total", ascending=False).head(top_n)


def region_passenger_totals(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Region", as_index=False)["Passengers_Total"].sum()


def yearly_trends(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Year", as_index=False)[
        ["Passengers_Total", "Freight_Total_(tonnes)", "Mail_Total_(tonnes)"]
    ].sum()


def plot_top_countries(df: pd.DataFrame, output_dir: Path, top_n: int = 10) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "top_countries.png"
    agg = top_countries_by_passengers(df, top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Passengers_Total", y="Country", data=agg, ax=ax)
    ax.set_title(f"Top {top_n} Countries by Total Passengers")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_region_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "region_distribution.png"
    agg = region_passenger_totals(df)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(agg["Passengers_Total"], labels=agg["Region"], autopct="%1.1f%%")
    ax.set_title("Passenger Traffic Share by Region")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_yearly_trends(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "yearly_trends.png"
    agg = yearly_trends(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x="Year", y="Passengers_Total", data=agg, label="Passengers", ax=ax)
    sns.lineplot(x="Year", y="Freight_Total_(tonnes)", data=agg, label="Freight", ax=ax)
    sns.lineplot(x="Year", y="Mail_Total_(tonnes)", data=agg, label="Mail", ax=ax)
    ax.set_title("Yearly Traffic Trends")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = [
    "plot_region_distribution",
    "plot_top_countries",
    "plot_yearly_trends",
    "region_passenger_totals",
    "top_countries_by_passengers",
    "yearly_trends",
]
