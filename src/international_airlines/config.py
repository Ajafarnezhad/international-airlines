"""Static configuration: paths, random state, and the country-to-region mapping."""

from __future__ import annotations

from dataclasses import dataclass

RANDOM_STATE = 42

NUMERIC_COLUMNS = [
    "Passengers_In",
    "Freight_In_(tonnes)",
    "Mail_In_(tonnes)",
    "Passengers_Out",
    "Freight_Out_(tonnes)",
    "Mail_Out_(tonnes)",
    "Passengers_Total",
    "Freight_Total_(tonnes)",
    "Mail_Total_(tonnes)",
]


@dataclass(frozen=True)
class ProjectPaths:
    raw_data_path: str = "data/city_pairs.csv"
    processed_data_path: str = "data/processed_city_pairs.csv"
    results_dir: str = "results"


DEFAULT_PATHS = ProjectPaths()

#: Maps a subset of the dataset's `Country` values to a broader region.
#: Any country not listed here falls back to "Other" (see
#: :func:`international_airlines.features.add_region`) rather than raising
#: or silently producing NaN.
REGION_MAP: dict[str, str] = {
    "New Zealand": "Oceania",
    "New..Zealand": "Oceania",  # a typo present in the source data
    "Bahrain": "Middle East",
    "India": "Asia",
    "Germany": "Europe",
    "UK": "Europe",
    "Oman": "Middle East",
    "Italy": "Europe",
    "Singapore": "Asia",
    "United Arab Emirates": "Middle East",
    "Thailand": "Asia",
    "Hong Kong": "Asia",
    "Hong Kong (SAR)": "Asia",
    "Solomon Islands": "Oceania",
    "USA": "North America",
    "Malaysia": "Asia",
    "Philippines": "Asia",
    "Fiji": "Oceania",
    "New Caledonia": "Oceania",
    "Papua New Guinea": "Oceania",
    "Vanuatu": "Oceania",
    "Japan": "Asia",
    "Canada": "North America",
    "Brunei": "Asia",
    "Indonesia": "Asia",
    "Netherlands": "Europe",
    "Greece": "Europe",
    "China": "Asia",
    "Yugoslavia": "Europe",
    "Sri Lanka": "Asia",
    "Cyprus": "Europe",
    "Mauritius": "Africa",
    "Nauru": "Oceania",
    "Tahiti": "Oceania",
    "Zimbabwe": "Africa",
    "South Africa": "Africa",
    "Western Samoa": "Oceania",
    "France": "Europe",
    "Denmark": "Europe",
    "Cook Islands": "Oceania",
    "American Samoa": "Oceania",
    "Austria": "Europe",
    "Argentina": "South America",
    "Guam": "Oceania",
    "Malta": "Europe",
    "Switzerland": "Europe",
    "Korea": "Asia",
    "Pakistan": "Asia",
    "Tonga": "Oceania",
    "Taiwan": "Asia",
    "Wallis Island": "Oceania",
    "Russia": "Europe",
    "Belgium": "Europe",
    "Lebanon": "Middle East",
    "Vietnam": "Asia",
    "Egypt": "Middle East",
    "Chile": "South America",
    "Luxembourg": "Europe",
    "East Timor": "Asia",
    "Niue": "Oceania",
    "Ireland": "Europe",
    "Macau": "Asia",
    "Brazil": "South America",
    "Kazakhstan": "Asia",
    "Marshall Islands": "Oceania",
    "Qatar": "Middle East",
    "Kenya": "Africa",
    "Nigeria": "Africa",
    "Kiribati": "Oceania",
    "Reunion": "Africa",
    "Turkey": "Europe",
    "Uruguay": "South America",
    "Laos": "Asia",
    "Peru": "South America",
    "Cambodia": "Asia",
    "Palau": "Oceania",
    "Azerbaijan": "Asia",
    "Botswana": "Africa",
}

__all__ = ["DEFAULT_PATHS", "NUMERIC_COLUMNS", "RANDOM_STATE", "REGION_MAP", "ProjectPaths"]
