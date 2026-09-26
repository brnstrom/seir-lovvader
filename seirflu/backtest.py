"""Rolling-origin backtests on training seasons (leave-one-season-out).

For a validation season the holiday/temperature effects and the priors for
the susceptible pool are re-estimated without that season, and weekly
forecasts are made from ISO week 40 to week 20 using only data up to each
cutoff. Used to choose forecast settings before looking at 2025/26.

Run from ``seir-model/``::

    python -m seirflu.backtest 2022 2023 2024
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .data import HUB_LOCATIONS
from .forecast import ForecastSettings, forecast_round
from .historical import TRAINING_SEASONS, estimate_effects, load_history, series_priors
from .variants import variant_config

RESULTS = Path(__file__).resolve().parents[1] / "results" / "backtest"


def reference_dates(season: int) -> list[date]:
    first = date.fromisocalendar(season, 40, 7)
    last = date.fromisocalendar(season + 1, 20, 7)
    out, day = [], first
    while day <= last:
        out.append(day)
        day += timedelta(days=7)
    return out


def backtest_season(season: int, variant: str = "main", weather_mode: str = "climatology",
                    n_samples: int = 1000, scales: tuple[float, ...] = (1.0, 0.5, 0.0)) -> pd.DataFrame:
    config = variant_config(variant)
    history = load_history()
    seasons = tuple(s for s in TRAINING_SEASONS if s != season)
    started = time.time()
    effects = estimate_effects(config, seasons=seasons, wide=history)
    prior, _ = series_priors(config, effects, seasons=seasons, wide=history)
    print(f"season {season}: effects re-estimated without it ({time.time() - started:.0f}s)", flush=True)
    settings = ForecastSettings(config=config, prior=prior, weather_mode=weather_mode,
                                n_samples=n_samples, future_rw_scales=scales)
    rows = []
    for ref in reference_dates(season):
        cutoff = ref - timedelta(days=7)
        wide = history[history.index <= pd.Timestamp(cutoff)]
        round_rows, _ = forecast_round(ref, wide, settings)
        rows.extend(round_rows)
        print(f"  [{time.time() - started:5.0f}s] {ref}", flush=True)
    table = pd.DataFrame(rows)
    table["season"] = season
    observed = history[list(HUB_LOCATIONS)].stack().rename("observed").reset_index()
    observed.columns = ["target_end_date", "location", "observed"]
    observed["target_end_date"] = observed["target_end_date"].dt.strftime("%Y-%m-%d")
    return table.merge(observed, on=["target_end_date", "location"], how="left")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seasons", nargs="+", type=int)
    parser.add_argument("--variant", default="main")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--scales", default="1,0.5,0")
    args = parser.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scales = tuple(float(v) for v in args.scales.split(","))
    for season in args.seasons:
        table = backtest_season(season, args.variant, n_samples=args.samples, scales=scales)
        table.to_csv(RESULTS / f"backtest_{args.variant}_{season}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
