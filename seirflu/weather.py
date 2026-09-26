"""Population-weighted daily weather per forecast series.

Weather enters the transmission model through the daily mean temperature.
Each county is represented by one SMHI station at its main population centre;
Skåne and Västra Götaland are split over several stations. County weights come
from SCB's 2024 municipal population totals.

For forecasting, weather after the data cutoff is never taken from the
observed record. It is replaced by a climatological expectation plus the most
recent anomaly decaying towards zero (see ``forecast_weather``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
GAP_FILL_CLIMATOLOGY_END = "2025-07-01"


@dataclass(frozen=True)
class WeatherPoint:
    name: str
    county_code: str
    share: float  # share of the county population represented by the point
    lat: float
    lon: float
    smhi_station: int  # SMHI station id, parameter 2 (daily mean temperature)


WEATHER_POINTS: tuple[WeatherPoint, ...] = (
    WeatherPoint("Stockholm", "01", 1.00, 59.33, 18.07, 98230),
    WeatherPoint("Uppsala", "03", 1.00, 59.86, 17.64, 97510),
    WeatherPoint("Eskilstuna", "04", 1.00, 59.37, 16.51, 96190),
    WeatherPoint("Linköping", "05", 1.00, 58.41, 15.62, 85240),
    WeatherPoint("Jönköping", "06", 1.00, 57.78, 14.16, 74460),
    WeatherPoint("Växjö", "07", 1.00, 56.88, 14.81, 64510),
    WeatherPoint("Kalmar", "08", 1.00, 56.66, 16.36, 66420),
    WeatherPoint("Visby", "09", 1.00, 57.64, 18.30, 78400),
    WeatherPoint("Karlskrona", "10", 1.00, 56.16, 15.59, 65090),
    WeatherPoint("Malmö", "12", 0.45, 55.60, 13.00, 52350),
    WeatherPoint("Helsingborg", "12", 0.30, 56.05, 12.69, 62040),
    WeatherPoint("Kristianstad", "12", 0.25, 56.03, 14.16, 64030),
    WeatherPoint("Halmstad", "13", 1.00, 56.67, 12.86, 62410),
    WeatherPoint("Göteborg", "14", 0.55, 57.71, 11.97, 71420),
    WeatherPoint("Borås", "14", 0.15, 57.72, 12.94, 72450),
    WeatherPoint("Trollhättan", "14", 0.15, 58.28, 12.29, 82230),
    WeatherPoint("Skövde", "14", 0.15, 58.39, 13.85, 83230),
    WeatherPoint("Karlstad", "17", 1.00, 59.38, 13.50, 93220),
    WeatherPoint("Örebro", "18", 1.00, 59.27, 15.21, 95130),
    WeatherPoint("Västerås", "19", 1.00, 59.61, 16.55, 96350),
    WeatherPoint("Falun", "20", 1.00, 60.61, 15.63, 105370),
    WeatherPoint("Gävle", "21", 1.00, 60.67, 17.14, 107420),
    WeatherPoint("Sundsvall", "22", 1.00, 62.39, 17.31, 127310),
    WeatherPoint("Östersund", "23", 1.00, 63.18, 14.64, 134110),
    WeatherPoint("Umeå", "24", 1.00, 63.83, 20.26, 140480),
    WeatherPoint("Luleå", "25", 1.00, 65.58, 22.15, 162860),
)

SERIES_COUNTIES = {
    "SE": None,  # all counties
    "SE-M": {"12"},
    "SE-O": {"14"},
    "SE-rest": "not-12-14",
}


def county_population() -> dict[str, int]:
    table = pd.read_csv(DATA / "scb_municipal_population.csv", dtype={"municipality_code": str})
    table = table[table["year"] == 2024]
    table["county_code"] = table["municipality_code"].str[:2]
    return table.groupby("county_code")["population"].sum().to_dict()


def series_point_weights(series: str) -> dict[str, float]:
    pops = county_population()
    selector = SERIES_COUNTIES[series]
    weights: dict[str, float] = {}
    for point in WEATHER_POINTS:
        code = point.county_code
        if selector is None:
            include = True
        elif selector == "not-12-14":
            include = code not in {"12", "14"}
        else:
            include = code in selector
        if include:
            weights[point.name] = weights.get(point.name, 0.0) + pops[code] * point.share
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


@lru_cache(maxsize=1)
def _points_table() -> pd.DataFrame:
    """Station temperatures, with gaps filled from the other stations.

    A missing station-day is replaced by that station's own harmonic
    climatology plus the mean anomaly of all stations reporting that day. The
    station climatologies used for the filling are fitted on data before
    ``GAP_FILL_CLIMATOLOGY_END`` so that no weather after the start of the
    2025/26 retrospective season enters earlier values.
    """
    table = pd.read_csv(DATA / "weather_points_daily.csv", parse_dates=["date"])
    wide = table.pivot(index="date", columns="point", values="temperature").sort_index()
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    doy = wide.index.dayofyear.to_numpy().astype(float)
    design = _harmonic_design(doy)
    clim = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    before = np.asarray(wide.index < pd.Timestamp(GAP_FILL_CLIMATOLOGY_END))
    for column in wide.columns:
        ok = wide[column].notna().to_numpy() & before
        coef, *_ = np.linalg.lstsq(design[ok], wide[column].to_numpy()[ok], rcond=None)
        clim[column] = design @ coef
    anomaly = (wide - clim).mean(axis=1, skipna=True)
    filled = wide.copy()
    for column in wide.columns:
        missing = filled[column].isna()
        filled.loc[missing, column] = clim.loc[missing, column] + anomaly[missing]
    return filled


@lru_cache(maxsize=8)
def series_weather(series: str) -> pd.DataFrame:
    """Daily population-weighted mean temperature (°C).

    Station gaps are filled in ``_points_table``; weights are renormalised if a
    station is still missing (days with under 80 % of the weight are dropped).
    """
    table = _points_table()
    weights = pd.Series(series_point_weights(series))
    values = table[weights.index]
    available = values.notna()
    weight_matrix = available.mul(weights, axis=1)
    covered = weight_matrix.sum(axis=1)
    temperature = (values.fillna(0.0) * weight_matrix).sum(axis=1) / covered.replace(0, np.nan)
    result = pd.DataFrame({"temperature": temperature})
    return result[covered >= 0.8].dropna()


def _harmonic_design(day_of_year: np.ndarray, n_harmonics: int = 3) -> np.ndarray:
    angle = 2 * np.pi * (day_of_year - 1) / 365.25
    columns = [np.ones_like(angle)]
    for k in range(1, n_harmonics + 1):
        columns.extend([np.cos(k * angle), np.sin(k * angle)])
    return np.column_stack(columns)


@lru_cache(maxsize=32)
def climatology_coefficients(series: str, variable: str, last_year: int) -> np.ndarray:
    """Harmonic climatology fitted to data before 1 July of ``last_year``.

    Using only earlier years keeps retrospective forecasts free of future
    weather information.
    """
    weather = series_weather(series)
    history = weather[weather.index < pd.Timestamp(f"{last_year}-07-01")]
    doy = history.index.dayofyear.to_numpy().astype(float)
    design = _harmonic_design(doy)
    coef, *_ = np.linalg.lstsq(design, history[variable].to_numpy(), rcond=None)
    return coef


def climatology(series: str, variable: str, dates: pd.DatetimeIndex, last_year: int) -> np.ndarray:
    coef = climatology_coefficients(series, variable, last_year)
    return _harmonic_design(dates.dayofyear.to_numpy().astype(float)) @ coef


def forecast_weather(
    series: str,
    variable: str,
    dates: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    *,
    mode: str = "climatology",
    anomaly_window: int = 7,
    anomaly_decay_days: float = 4.0,
) -> np.ndarray:
    """Weather covariate on ``dates`` as it would be known at ``cutoff``.

    ``mode``:
      * ``climatology``: observed up to the cutoff; afterwards climatology plus
        the mean anomaly of the last ``anomaly_window`` days, decaying with
        e-folding time ``anomaly_decay_days``.
      * ``oracle``: observed values throughout (uses future information; only
        for measuring the value of a perfect weather forecast).
    """
    weather = series_weather(series)[variable]
    last_year = cutoff.year if cutoff.month >= 7 else cutoff.year - 1
    clim = climatology(series, variable, dates, last_year)
    observed = weather.reindex(dates).to_numpy(copy=True)
    if mode == "oracle":
        missing = np.isnan(observed)
        observed[missing] = clim[missing]
        return observed
    if mode != "climatology":
        raise ValueError(f"Unknown weather forecast mode {mode!r}")
    past = dates <= cutoff
    values = np.where(past, observed, np.nan)
    recent_dates = pd.date_range(cutoff - pd.Timedelta(days=anomaly_window - 1), cutoff)
    recent_obs = weather.reindex(recent_dates).to_numpy()
    recent_clim = climatology(series, variable, recent_dates, last_year)
    anomaly = np.nanmean(recent_obs - recent_clim) if np.isfinite(recent_obs).any() else 0.0
    lead = (dates - cutoff).days.to_numpy().astype(float)
    future = ~past
    values[future] = clim[future] + anomaly * np.exp(-lead[future] / anomaly_decay_days)
    gaps = np.isnan(values)
    values[gaps] = clim[gaps]
    return values
