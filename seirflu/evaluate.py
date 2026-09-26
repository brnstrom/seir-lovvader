"""Score retrospective forecasts against the fixed 2025/26 outcomes.

Point forecasts are scored with MAE, RMSE and bias (forecast minus observed),
as in the hub's evaluation pipeline. Predictive quantiles are scored with the
weighted interval score (Bracher et al. 2021) over the 50/80/95 % intervals,
and compared with the hub's probabilistic normal-MA3 baseline on the same
tasks.

Run from ``seir-model/``::

    python -m seirflu.evaluate main no-holidays no-temperature ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FINAL_OUTCOMES, HUB_ROOT

RESULTS = Path(__file__).resolve().parents[1] / "results" / "retrospective"
BASELINE_MODELS = ("hubdemo-persistence", "hubdemo-mean3", "hubdemo-mean6",
                   "hubdemo-linear4", "hubdemo-damped4", "hubdemo-exptrend4")
INTERVALS = ((0.25, 0.75, 0.5), (0.1, 0.9, 0.2), (0.025, 0.975, 0.05))


def outcomes() -> pd.DataFrame:
    table = pd.read_csv(FINAL_OUTCOMES)
    table = table[table["status"] == "available"]
    return table[["location", "target_end_date", "value"]].rename(columns={"value": "observed"})


def load_model_forecasts(tag: str) -> pd.DataFrame:
    table = pd.read_csv(RESULTS / f"forecasts_{tag}.csv")
    if "future_rw_scale" in table:
        table["model"] = tag + "@rw" + table["future_rw_scale"].map(lambda v: f"{v:g}")
    else:
        table["model"] = tag
    return table


def load_hub_model(model_id: str) -> pd.DataFrame:
    frames = []
    for path in sorted((HUB_ROOT / "model-output" / model_id).glob("*.csv")):
        frame = pd.read_csv(path)
        frames.append(frame)
    table = pd.concat(frames)
    table["target_end_date"] = (
        pd.to_datetime(table["reference_date"]) + pd.to_timedelta(7 * table["horizon"], unit="D")
    ).dt.strftime("%Y-%m-%d")
    table = table.rename(columns={"value": "mean"})
    table["model"] = model_id
    return table[["model", "reference_date", "location", "horizon", "target_end_date", "mean"]]


def wis(row: pd.Series) -> float:
    y = row["observed"]
    total = 0.5 * abs(y - row["q0.5"])
    for lower_q, upper_q, alpha in INTERVALS:
        lower, upper = row[f"q{lower_q:g}"], row[f"q{upper_q:g}"]
        score = (upper - lower) + (2 / alpha) * max(lower - y, 0) + (2 / alpha) * max(y - upper, 0)
        total += alpha / 2 * score
    return total / (len(INTERVALS) + 0.5)


def point_metrics(table: pd.DataFrame, value: str, by=("model", "location", "horizon")) -> pd.DataFrame:
    err = table[value] - table["observed"]
    frame = table.assign(error=err, abs_error=err.abs(), sq_error=err**2)
    grouped = frame.groupby(list(by))
    return pd.DataFrame({
        "n": grouped.size(),
        "mae": grouped["abs_error"].mean(),
        "rmse": np.sqrt(grouped["sq_error"].mean()),
        "bias": grouped["error"].mean(),
    }).reset_index()


def normal_ma3_baseline() -> pd.DataFrame:
    path = HUB_ROOT / "evaluation-output" / "2025-2026" / "normal-ma3-baseline-forecasts.csv"
    table = pd.read_csv(path)
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args(argv)
    obs = outcomes()
    frames = []
    for tag in args.tags:
        frames.append(load_model_forecasts(tag))
    ours = pd.concat(frames).merge(obs, on=["location", "target_end_date"], how="inner")
    baselines = pd.concat([load_hub_model(m) for m in BASELINE_MODELS])
    baselines = baselines.merge(obs, on=["location", "target_end_date"], how="inner")

    rows = []
    for value in ("mean", "q0.5", "map_mean"):
        metrics = point_metrics(ours, value)
        metrics["point"] = value
        rows.append(metrics)
    base = point_metrics(baselines, "mean")
    base["point"] = "mean"
    rows.append(base)
    table = pd.concat(rows)
    table.to_csv(RESULTS / "point_metrics.csv", index=False)

    pivot = table[table["point"] == "mean"].pivot_table(index=["location", "horizon"], columns="model", values="mae")
    pd.set_option("display.width", 250)
    print("MAE of the Laplace posterior predictive mean (baselines: their submitted value):")
    print(pivot.round(1).to_string())
    for value in ("q0.5", "map_mean"):
        alt = table[table["point"] == value].pivot_table(index=["location", "horizon"], columns="model", values="mae")
        print(f"\nMAE if the point forecast were {value}:")
        print(alt.round(1).to_string())
    bias = table[table["point"] == "map_mean"].pivot_table(index=["location", "horizon"], columns="model", values="bias")
    print("\nBias (forecast - observed) of the submitted point forecast (expected count at the posterior mode):")
    print(bias.round(1).to_string())

    ours["wis"] = ours.apply(wis, axis=1)
    for lower_q, upper_q, alpha in INTERVALS:
        name = f"cov{int(round(100 * (1 - alpha)))}"
        ours[name] = ((ours["observed"] >= ours[f"q{lower_q:g}"]) & (ours["observed"] <= ours[f"q{upper_q:g}"])).astype(float)
    ours.to_csv(RESULTS / "scored_forecasts.csv", index=False)
    summary = ours.groupby(["model", "location", "horizon"])[["wis", "cov50", "cov80", "cov95"]].mean()
    print("\nWIS and interval coverage:")
    print(summary.round(2).unstack("model").to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
