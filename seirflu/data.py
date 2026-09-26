"""Reading hub target data and building the weekly series used by the model."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# The hub repository (target data, round files, model-output). By default the
# model directory sits inside a hub checkout; SEIRFLU_HUB_ROOT points
# elsewhere, e.g. when the model code lives in its own repository.
HUB_ROOT = Path(os.environ.get("SEIRFLU_HUB_ROOT", Path(__file__).resolve().parents[2])).resolve()
ROUND_ROOT = HUB_ROOT / "retrospective-data" / "2025-2026" / "rounds"
MANIFEST = HUB_ROOT / "retrospective-data" / "2025-2026" / "manifest.csv"
FINAL_OUTCOMES = HUB_ROOT / "retrospective-data" / "2025-2026" / "final-outcomes.csv"
LIVE_TARGET = HUB_ROOT / "target-data" / "time-series.csv"

HUB_LOCATIONS = ("SE", "SE-M", "SE-O")
SEASON_START_WEEK = 30  # ISO week in which a season (and its observations) starts
BURN_IN_WEEKS = 4  # the model window starts this many weeks earlier, unobserved


def read_target_file(path: Path, cutoff: date | None = None) -> pd.DataFrame:
    """Return a wide weekly table (index: week-ending Sunday, columns: locations).

    Suppressed or missing values are NaN. Rows after ``cutoff`` are dropped.
    """
    table = pd.read_csv(path, dtype={"value": "string"})
    table["target_end_date"] = pd.to_datetime(table["target_end_date"])
    if cutoff is not None:
        table = table[table["target_end_date"] <= pd.Timestamp(cutoff)]
    table["value"] = pd.to_numeric(table["value"], errors="coerce").astype("float64")
    table.loc[table["status"] != "available", "value"] = np.nan
    wide = table.pivot(index="target_end_date", columns="location", values="value").sort_index()
    wide = wide.reindex(columns=list(HUB_LOCATIONS))
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="W-SUN")
    return wide.reindex(full_index).astype("float64")


def add_rest_of_sweden(wide: pd.DataFrame) -> pd.DataFrame:
    """Add ``SE-rest`` = Sweden minus Skåne minus Västra Götaland."""
    out = wide.copy()
    out["SE-rest"] = wide["SE"] - wide["SE-M"] - wide["SE-O"]
    out.loc[out["SE-rest"] < 0, "SE-rest"] = np.nan
    return out


def season_of(day: date) -> int:
    """Season label (first calendar year) for a date; seasons start in ISO week 30."""
    iso_year, iso_week, _ = day.isocalendar()
    return iso_year if iso_week >= SEASON_START_WEEK else iso_year - 1


def season_start(season: int) -> date:
    """Monday of ISO week 30 of the season's first year."""
    return date.fromisocalendar(season, SEASON_START_WEEK, 1)


def window_start(season: int) -> date:
    """Start of the model window: a few unobserved burn-in weeks before week 30.

    The burn-in lets the initial condition (a constant infection history)
    settle before the first observation enters the likelihood.
    """
    return season_start(season) - timedelta(days=7 * BURN_IN_WEEKS)


def season_end(season: int) -> date:
    """Sunday of ISO week 29 of the following year."""
    return date.fromisocalendar(season + 1, SEASON_START_WEEK, 1) - timedelta(days=1)


def manifest() -> pd.DataFrame:
    table = pd.read_csv(MANIFEST)
    table["reference_date"] = pd.to_datetime(table["reference_date"]).dt.date
    table["data_cutoff"] = pd.to_datetime(table["data_cutoff"]).dt.date
    return table


def final_outcomes() -> pd.DataFrame:
    return read_target_file(FINAL_OUTCOMES)
