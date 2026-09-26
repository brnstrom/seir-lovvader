"""Make a live forecast for one hub round and write the submission file.

The current season is read from ``target-data/time-series.csv`` in the hub
checkout. Holiday and temperature effects come from the prior file (by default
the one estimated on seasons up to 2024/25; ``--prior`` can point to a refit
that includes 2025/26). Refresh the weather cache first so that the
temperature record reaches the data cutoff::

    python scripts/fetch_weather.py      # SMHI, takes a minute
    python -m seirflu.live 2026-10-04    # from the model directory

``python -m seirflu.weekly`` does both and checks that the round's data are
in place. The school calendar for 2026/27 is already in ``data/``; add the
next school year's Skolporten file to ``data/raw/`` and rerun
``scripts/parse_school_calendars.py`` before the 2027/28 season.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .data import HUB_ROOT, LIVE_TARGET, read_target_file
from .forecast import ForecastSettings, forecast_round
from .historical import prior_path
from .retrospective import load_prior, write_hub_files
from .variants import variant_config
from .weather import series_weather

DEFAULT_MODEL_ID = "brannstrom-seirlovvader"


def weather_gaps(cutoff: date) -> list[str]:
    """Series whose temperature record ends before the data cutoff."""
    messages = []
    for series in ("SE", "SE-M", "SE-O"):
        last = series_weather(series).index.max().date()
        if last < cutoff:
            messages.append(f"temperature for {series} ends {last}; days up to the cutoff {cutoff} "
                            "use climatology")
    return messages


def run_live(reference_date: date, model_id: str = DEFAULT_MODEL_ID, input_path: Path = LIVE_TARGET,
             prior: Path | None = None, variant: str = "main") -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Forecast one round; return (hub file, forecast table, diagnostics)."""
    if reference_date.weekday() != 6:
        raise ValueError("reference_date must be a Sunday")
    cutoff = reference_date - timedelta(days=7)
    wide = read_target_file(input_path, cutoff=cutoff)
    config = variant_config(variant)
    settings = ForecastSettings(config=config, prior=load_prior(prior or prior_path(variant)),
                                future_rw_scales=(1.0,))
    rows, diagnostics = forecast_round(reference_date, wide, settings)
    write_hub_files(rows, model_id, 1.0, "map_mean")
    path = HUB_ROOT / "model-output" / model_id / f"{reference_date.isoformat()}-{model_id}.csv"
    table = pd.DataFrame(rows)[["location", "horizon", "target_end_date", "map_mean", "q0.1", "q0.5", "q0.9"]]
    return path, table, pd.DataFrame(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_date", help="Sunday of the forecast round, YYYY-MM-DD")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--input", type=Path, default=LIVE_TARGET)
    parser.add_argument("--prior", type=Path, default=None)
    parser.add_argument("--variant", default="main")
    args = parser.parse_args()

    reference_date = date.fromisoformat(args.reference_date)
    for message in weather_gaps(reference_date - timedelta(days=7)):
        print(f"Warning: {message}. Run scripts/fetch_weather.py first.")
    path, table, diagnostics = run_live(reference_date, args.model_id, args.input, args.prior, args.variant)
    print(table.round(1).to_string(index=False))
    print(diagnostics[["location", "r_eff_cutoff", "r_eff_cutoff_q10", "r_eff_cutoff_q90"]].round(2).to_string(index=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
