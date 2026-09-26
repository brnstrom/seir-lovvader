"""Weekly live round: check the data, refresh the weather, forecast, write the file.

This is the entry point of the scheduled workflow
(``.github/workflows/weekly-forecast.yml``) and can also be run by hand from
the model directory::

    python -m seirflu.weekly                  # the coming Sunday's round
    python -m seirflu.weekly 2026-10-04 --skip-weather

The hub publishes the week that ends on ``reference_date - 7`` (the data
cutoff), usually on Thursday or Friday. If that week is not yet in
``target-data/time-series.csv`` the run stops with exit code 2 and writes
nothing, unless ``--allow-missing-last-week`` is given (the scheduled Sunday
run uses it so that a late data release does not cost the round; the model
then treats the week as unobserved). Rounds outside the live season (ISO
weeks 40 to 20) are skipped with exit code 0. A failed SMHI download is
reported but does not stop the run; the cached temperatures are used.

The run writes ``model-output/<model_id>/<reference_date>-<model_id>.csv``
in the hub checkout, a Markdown summary for the pull request under
``results/live/`` and, with ``--github-output``, the step outputs ``skip``,
``reference_date``, ``file`` (relative to the hub root) and ``summary``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .data import HUB_LOCATIONS, HUB_ROOT, LIVE_TARGET

MODEL_ROOT = Path(__file__).resolve().parents[1]
LIVE_RESULTS = MODEL_ROOT / "results" / "live"
DATA_NOT_READY = 2
LOCATION_NAMES = {"SE": "Sweden", "SE-M": "Skåne", "SE-O": "Västra Götaland"}


def coming_sunday(today: date) -> date:
    return today + timedelta(days=6 - today.weekday())


def in_live_season(reference_date: date) -> bool:
    """Hub rounds run from ISO week 40 to ISO week 20."""
    week = reference_date.isocalendar()[1]
    return week >= 40 or week <= 20


def write_outputs(path: Path | None, **values: str) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def missing_locations(cutoff: date, path: Path = LIVE_TARGET) -> list[str]:
    """Locations without a published row for the week that ends on ``cutoff``."""
    table = pd.read_csv(path, dtype={"value": "string"})
    published = set(table.loc[table["target_end_date"] == cutoff.isoformat(), "location"])
    return [loc for loc in HUB_LOCATIONS if loc not in published]


def summary_markdown(reference_date: date, model_id: str, table: pd.DataFrame, diagnostics: pd.DataFrame,
                     notes: list[str]) -> str:
    lines = [
        f"Forecast from `{model_id}` for the round with reference date {reference_date.isoformat()} "
        f"(data up to the week ending {(reference_date - timedelta(days=7)).isoformat()}).",
        "",
        "| Location | Week ending | Horizon | Expected cases | 80 % interval |",
        "| --- | --- | --- | --- | --- |",
    ]
    named = table.rename(columns={"q0.1": "q10", "q0.9": "q90"})
    for row in named.itertuples(index=False):
        lines.append(f"| {LOCATION_NAMES.get(row.location, row.location)} | {row.target_end_date} | {row.horizon} "
                     f"| {row.map_mean:.0f} | {row.q10:.0f} to {row.q90:.0f} |")
    lines += ["", "Effective reproduction number at the data cutoff (80 % interval):", ""]
    for row in diagnostics.itertuples(index=False):
        lines.append(f"- {LOCATION_NAMES.get(row.location, row.location)}: {row.r_eff_cutoff:.2f} "
                     f"({row.r_eff_cutoff_q10:.2f} to {row.r_eff_cutoff_q90:.2f})")
    if notes:
        lines += ["", *[f"Note: {note}." for note in notes]]
    lines += ["", "Generated automatically by the model's weekly workflow."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_date", nargs="?", default="",
                        help="Sunday of the round (YYYY-MM-DD); default: the coming Sunday in Stockholm")
    parser.add_argument("--model-id", default="brannstrom-seirlovvader")
    parser.add_argument("--skip-weather", action="store_true", help="do not download SMHI data first")
    parser.add_argument("--allow-missing-last-week", action="store_true",
                        help="forecast even if the cutoff week is not yet published")
    parser.add_argument("--github-output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.reference_date:
        reference_date = date.fromisoformat(args.reference_date)
    else:
        reference_date = coming_sunday(datetime.now(ZoneInfo("Europe/Stockholm")).date())
    if reference_date.weekday() != 6:
        raise SystemExit("reference_date must be a Sunday")
    cutoff = reference_date - timedelta(days=7)
    if not in_live_season(reference_date):
        print(f"{reference_date} is outside the live season (ISO weeks 40 to 20); nothing to do.")
        write_outputs(args.github_output, skip="true")
        return 0

    notes = []
    missing = missing_locations(cutoff)
    if missing and not args.allow_missing_last_week:
        print(f"The week ending {cutoff} is not yet published for {', '.join(missing)} "
              f"in {LIVE_TARGET}; nothing written.")
        return DATA_NOT_READY
    if missing:
        notes.append(f"the week ending {cutoff} was not yet published for {', '.join(missing)}; "
                     "the forecast uses the data available")

    if not args.skip_weather:
        fetched = subprocess.run([sys.executable, str(MODEL_ROOT / "scripts" / "fetch_weather.py")])
        if fetched.returncode != 0:
            notes.append("the SMHI download failed; cached temperatures and climatology were used")

    from .live import run_live, weather_gaps  # imported late so the refreshed weather is read

    notes += weather_gaps(cutoff)
    path, table, diagnostics = run_live(reference_date, args.model_id)
    print(table.round(1).to_string(index=False))
    LIVE_RESULTS.mkdir(parents=True, exist_ok=True)
    summary = summary_markdown(reference_date, args.model_id, table, diagnostics, notes)
    summary_path = LIVE_RESULTS / f"pr-body-{reference_date.isoformat()}.md"
    summary_path.write_text(summary, encoding="utf-8")
    table.to_csv(LIVE_RESULTS / f"forecast-{reference_date.isoformat()}.csv", index=False)
    diagnostics.to_csv(LIVE_RESULTS / f"diagnostics-{reference_date.isoformat()}.csv", index=False)
    print(summary)
    write_outputs(args.github_output, skip="false", reference_date=reference_date.isoformat(),
                  file=path.relative_to(HUB_ROOT).as_posix(), summary=str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
