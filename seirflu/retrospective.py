"""Run the 33 retrospective 2025/26 rounds and write hub submission files.

For every manifest row the model reads only the named round file, cut at the
round's data cutoff. Holiday and weather effects come from
``results/historical/forecast_prior_<variant>.json`` (estimated on seasons up
to 2024/25). Weather after the cutoff is climatology plus a decaying anomaly
unless ``--weather oracle`` is given.

Run from ``seir-model/``::

    python -m seirflu.retrospective --model-id brannstrom-seirlovvader
    python -m seirflu.retrospective --variant no-holidays --no-hub-output
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

from .data import HUB_ROOT, ROUND_ROOT, manifest, read_target_file
from .fit import SharedPrior
from .forecast import ForecastSettings, forecast_round
from .historical import prior_path
from .variants import variant_config

RESULTS = Path(__file__).resolve().parents[1] / "results" / "retrospective"
TARGET = "weekly incident influenza cases"


def load_prior(path: Path) -> SharedPrior:
    with path.open() as handle:
        return SharedPrior.from_dict(json.load(handle))


def write_hub_files(rows: list[dict], model_id: str, future_rw_scale: float, point: str = "map_mean") -> None:
    output_root = HUB_ROOT / "model-output" / model_id
    output_root.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table = table[table["future_rw_scale"] == future_rw_scale]
    order = {"SE": 0, "SE-M": 1, "SE-O": 2}
    for reference_date, group in table.groupby("reference_date"):
        group = group.sort_values(["location", "horizon"], key=lambda s: s.map(order) if s.name == "location" else s)
        path = output_root / f"{reference_date}-{model_id}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["reference_date", "target", "horizon", "location", "output_type", "output_type_id", "value"])
            for row in group.itertuples(index=False):
                value = getattr(row, point)
                writer.writerow([reference_date, TARGET, row.horizon, row.location, "mean", "", f"{value:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="brannstrom-seirlovvader")
    parser.add_argument("--variant", default="main")
    parser.add_argument("--weather", default="climatology", choices=["climatology", "oracle"])
    parser.add_argument("--prior", type=Path, default=None, help="defaults to the variant's prior file")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=None, help="only the first N rounds (testing)")
    parser.add_argument("--no-hub-output", action="store_true")
    parser.add_argument("--tag", default=None, help="name for the results files")
    parser.add_argument("--future-rw", type=float, default=None,
                        help="post-cutoff random-walk scale used for hub files (default: first scale)")
    parser.add_argument("--point", default="map_mean", choices=["map_mean", "mean"],
                        help="hub point forecast: expected count at the posterior mode (default) "
                             "or the Laplace posterior predictive mean")
    parser.add_argument("--scales", default="1,0.5,0", help="comma-separated post-cutoff random-walk scales")
    parser.add_argument("--from-results", action="store_true",
                        help="only rewrite hub files from an existing forecasts_<tag>.csv")
    args = parser.parse_args()

    config = variant_config(args.variant)
    prior = load_prior(args.prior or prior_path(args.variant))
    scales = tuple(float(v) for v in args.scales.split(","))
    settings = ForecastSettings(config=config, prior=prior, weather_mode=args.weather,
                                n_samples=args.samples, future_rw_scales=scales)
    tag = args.tag or (args.variant if args.weather == "climatology" else f"{args.variant}-oracle-weather")
    if args.from_results:
        rows = pd.read_csv(RESULTS / f"forecasts_{tag}.csv").to_dict("records")
        scale = scales[0] if args.future_rw is None else args.future_rw
        write_hub_files(rows, args.model_id, scale, args.point)
        print(f"Rewrote hub files for model-output/{args.model_id}/ from forecasts_{tag}.csv ({args.point})")
        return
    rounds = manifest()
    if args.rounds:
        rounds = rounds.head(args.rounds)
    all_rows, all_diag = [], []
    started = time.time()
    for row in rounds.itertuples(index=False):
        wide = read_target_file(ROUND_ROOT / f"{row.reference_date.isoformat()}.csv", cutoff=row.data_cutoff)
        rows, diagnostics = forecast_round(row.reference_date, wide, settings)
        all_rows.extend(rows)
        all_diag.extend(diagnostics)
        first = rows[0]["future_rw_scale"]
        shown = [r for r in rows if r["future_rw_scale"] == first]
        summary = "  ".join(f"{r['location']} h0={r['mean']:.0f} h3={shown[i + 3]['mean']:.0f}"
                            for i, r in enumerate(shown) if r["horizon"] == 0)
        print(f"[{time.time() - started:6.0f}s] {row.reference_date}: {summary}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(RESULTS / f"forecasts_{tag}.csv", index=False)
    pd.DataFrame(all_diag).to_csv(RESULTS / f"diagnostics_{tag}.csv", index=False)
    if not args.no_hub_output and args.variant == "main" and args.weather == "climatology":
        scale = settings.future_rw_scales[0] if args.future_rw is None else args.future_rw
        write_hub_files(all_rows, args.model_id, scale, args.point)
        print(f"Wrote hub files to model-output/{args.model_id}/")


if __name__ == "__main__":
    main()
