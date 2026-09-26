"""Score backtests on training seasons against simple baselines."""

from __future__ import annotations

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .data import HUB_LOCATIONS, HUB_ROOT
from .historical import load_history

RESULTS = Path(__file__).resolve().parents[1] / "results" / "backtest"


def demo_models():
    path = HUB_ROOT / "demo-models" / "generate_demo_forecasts.py"
    spec = importlib.util.spec_from_file_location("demo_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MODELS


def baseline_forecasts(reference_dates, history: pd.DataFrame) -> pd.DataFrame:
    models = demo_models()
    rows = []
    for ref in reference_dates:
        cutoff = pd.Timestamp(ref) - pd.Timedelta(days=7)
        for loc in HUB_LOCATIONS:
            series = history.loc[history.index <= cutoff, loc].dropna().tolist()
            for name, model in models.items():
                for h in range(4):
                    rows.append({
                        "model": name, "reference_date": pd.Timestamp(ref).strftime("%Y-%m-%d"),
                        "location": loc, "horizon": h,
                        "target_end_date": (pd.Timestamp(ref) + pd.Timedelta(days=7 * h)).strftime("%Y-%m-%d"),
                        "point": model(series, h + 1),
                    })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    tag = argv[0] if argv else "main"
    frames = [pd.read_csv(p) for p in sorted(RESULTS.glob(f"backtest_{tag}_*.csv"))]
    table = pd.concat(frames)
    history = load_history()
    obs = history[list(HUB_LOCATIONS)].stack().rename("observed").reset_index()
    obs.columns = ["target_end_date", "location", "observed"]
    obs["target_end_date"] = obs["target_end_date"].dt.strftime("%Y-%m-%d")
    base = baseline_forecasts(sorted(table["reference_date"].unique()), history)
    base = base.merge(obs, on=["target_end_date", "location"])
    base = base.merge(table[["reference_date", "season"]].drop_duplicates(), on="reference_date")
    long = []
    for scale, group in table.groupby("future_rw_scale"):
        for col, label in (("mean", "mean"), ("q0.5", "median"), ("map_mean", "map")):
            g = group[["season", "reference_date", "location", "horizon", "target_end_date", "observed", col]].rename(columns={col: "point"})
            g["model"] = f"seir-{label}@rw{scale:g}"
            long.append(g)
    long.append(base)
    everything = pd.concat(long).dropna(subset=["observed"])
    everything["abs_error"] = (everything["point"] - everything["observed"]).abs()
    everything["sq_error"] = (everything["point"] - everything["observed"]) ** 2
    pd.set_option("display.width", 250)
    for loc in HUB_LOCATIONS:
        sub = everything[everything["location"] == loc]
        mae = sub.pivot_table(index=["season", "horizon"], columns="model", values="abs_error", aggfunc="mean")
        print(f"\n=== {loc}: MAE by season and horizon")
        keep = [c for c in mae.columns if c.startswith("seir") and ("rw1" in c or "rw0@" in c or c.endswith("rw0"))] + ["hubdemo-persistence", "hubdemo-damped4", "hubdemo-mean3"]
        print(mae[keep].round(1).to_string())
    overall = everything.groupby(["model", "location"])["abs_error"].mean().unstack("location")
    overall["relative_to_persistence"] = (overall / overall.loc["hubdemo-persistence"]).mean(axis=1)
    print("\n=== Mean absolute error over all seasons and horizons")
    print(overall.round(2).sort_values("relative_to_persistence").to_string())
    rmse = np.sqrt(everything.groupby(["model", "location"])["sq_error"].mean()).unstack("location")
    print("\n=== RMSE over all seasons and horizons")
    print(rmse.round(1).loc[overall.sort_values("relative_to_persistence").index].to_string())
    everything.to_csv(RESULTS / f"scored_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
