"""Estimate holiday, temperature and reporting effects from historical seasons.

The effects are estimated jointly from three non-overlapping series (Region
Skåne, Västra Götaland and the rest of Sweden) over the training seasons
2015/16-2024/25, excluding 2020/21 and 2021/22 (COVID-19). Only data available
before the first 2025/26 forecast round are used, so the estimates can be used
in every retrospective round without look-ahead.

Run from ``seir-model/``::

    python -m seirflu.historical
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .data import HUB_LOCATIONS, ROUND_ROOT, add_rest_of_sweden, read_target_file
from .fit import Problem, SharedPrior, shared_names
from .model import HOLIDAY_TYPES, ModelConfig, generation_summary
from .units import EXCLUDED_SEASONS, historical_units
from .variants import MAIN_CONFIG, variant_config

RESULTS = Path(__file__).resolve().parents[1] / "results" / "historical"
TRAINING_SEASONS = tuple(s for s in range(2015, 2025) if s not in EXCLUDED_SEASONS)
ESTIMATION_SERIES = ("SE-M", "SE-O", "SE-rest")
HISTORY_FILE = ROUND_ROOT / "2025-10-05.csv"
HISTORY_CUTOFF = date(2025, 9, 28)


def load_history() -> pd.DataFrame:
    return add_rest_of_sweden(read_target_file(HISTORY_FILE, cutoff=HISTORY_CUTOFF))


def estimate_effects(config: ModelConfig, seasons=TRAINING_SEASONS, series=ESTIMATION_SERIES,
                     wide: pd.DataFrame | None = None, verbose: bool = False) -> dict:
    wide = load_history() if wide is None else wide
    units = historical_units(wide, series, tuple(seasons), config)
    problem = Problem(units, config)
    started = time.time()
    result = problem.fit(verbose=verbose)
    hess, chol, log_evidence = problem.laplace(result.x)
    cov = np.linalg.inv(hess)
    idx = problem.shared_indices()
    theta = problem.shared_vector(result.x)
    cov_theta = cov[np.ix_(idx, idx)]
    loglik = float(problem.log_likelihood_terms(result.x).sum())
    return {
        "theta": theta,
        "cov": cov_theta,
        "se": np.sqrt(np.diag(cov_theta)),
        "log_evidence": float(log_evidence),
        "loglik": loglik,
        "flat": result.x,
        "grad_max": float(np.max(np.abs(result.jac))),
        "problem": problem,
        "seconds": time.time() - started,
    }


def series_priors(config: ModelConfig, effects: dict, seasons=TRAINING_SEASONS,
                  wide: pd.DataFrame | None = None, recent_from: int = 2022) -> tuple[SharedPrior, pd.DataFrame]:
    """Fit the hub locations with the effects fixed by their posterior.

    Returns the forecasting prior: effects ~ N(theta, cov); per location a
    log-normal prior for the susceptible pool K (from post-COVID seasons,
    whose testing volume matches the current era) and for the dispersion.
    """
    wide = load_history() if wide is None else wide
    units = historical_units(wide, HUB_LOCATIONS, tuple(seasons), config)
    base = SharedPrior(
        mean=effects["theta"], cov=effects["cov"],
        log_phi={s: config.log_phi_prior for s in HUB_LOCATIONS},
    )
    problem = Problem(units, config, shared_prior=base, series_names=list(HUB_LOCATIONS))
    result = problem.fit()
    params = problem.params(result.x)
    rows = []
    for i, unit in enumerate(units):
        observed = unit.y[unit.obs > 0]
        rows.append({
            "series": unit.series,
            "season": unit.season,
            "log_k": float(params["units"]["log_k"][i]),
            "k": float(np.exp(params["units"]["log_k"][i])),
            "season_total": float(observed.sum()),
            "season_peak": float(observed.max()) if len(observed) else np.nan,
        })
    table = pd.DataFrame(rows)
    log_k = {}
    for series in HUB_LOCATIONS:
        recent = table[(table["series"] == series) & (table["season"] >= recent_from)]["log_k"]
        allseasons = table[table["series"] == series]["log_k"]
        spread = max(float(allseasons.std(ddof=1)), 0.5)
        log_k[series] = (float(recent.mean()), spread)
    log_phi = {
        series: (float(params["shared"]["log_phi"][i]), 0.5)
        for i, series in enumerate(problem.series_names)
    }
    prior = SharedPrior(mean=effects["theta"], cov=effects["cov"], log_phi=log_phi, log_k=log_k)
    return prior, table


def leave_one_season_out(config: ModelConfig, wide: pd.DataFrame | None = None) -> pd.DataFrame:
    wide = load_history() if wide is None else wide
    rows = []
    for left_out in TRAINING_SEASONS:
        seasons = tuple(s for s in TRAINING_SEASONS if s != left_out)
        fit = estimate_effects(config, seasons=seasons, wide=wide)
        row = {"left_out_season": f"{left_out}/{str(left_out + 1)[2:]}"}
        row.update({name: value for name, value in zip(shared_names(), fit["theta"])})
        row.update({f"se_{name}": value for name, value in zip(shared_names(), fit["se"])})
        rows.append(row)
        print(f"  without {left_out}: " + ", ".join(f"{n}={v:+.3f}" for n, v in zip(shared_names(), fit["theta"])))
    return pd.DataFrame(rows)


def fitted_table(effects: dict) -> pd.DataFrame:
    problem: Problem = effects["problem"]
    weekly, x, r_t, r_eff = problem.simulate_flat(effects["flat"])
    weekly, r_t, r_eff = np.asarray(weekly), np.asarray(r_t), np.asarray(r_eff)
    rows = []
    for i, unit in enumerate(problem.units):
        ends = unit.week_ends
        for w in range(unit.n_weeks):
            days = slice(7 * w, 7 * w + 7)
            rows.append({
                "series": unit.series,
                "season": unit.season,
                "week_end": ends[w].isoformat(),
                "observed": unit.y[w] if unit.obs[w] > 0 else np.nan,
                "fitted": weekly[i, w],
                "r_t": float(r_t[i, days].mean()),
                "r_eff": float(r_eff[i, days].mean()),
                "temperature": float(unit.temp[days].mean()),
                **{f"hol_{h}": float(unit.hol[days, k].mean()) for k, h in enumerate(HOLIDAY_TYPES)},
            })
    return pd.DataFrame(rows)


TEMPERATURE_GRID = (-15.0, -10.0, -5.0, 0.0, 10.0, 15.0, 20.0)


def temperature_multiplier(theta: np.ndarray, temps, form: str) -> np.ndarray:
    """R at temperature T relative to R at 5 °C, for one parameter vector."""
    temps = np.asarray(temps, dtype=float)
    slope_or_amp, mid, log_width = theta[5], theta[6], theta[7]
    if form == "linear":
        return np.exp(slope_or_amp * (temps - 5.0))
    if form == "sigmoid":
        width = np.exp(log_width)
        logistic = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        return np.exp(slope_or_amp * (logistic((mid - temps) / width) - logistic((mid - 5.0) / width)))
    return np.ones_like(temps)


def effects_table(effects: dict, config: ModelConfig) -> pd.DataFrame:
    rows = []
    for name, value, se in zip(shared_names(), effects["theta"], effects["se"]):
        if name in ("temp_mid", "temp_log_width") and config.temperature != "sigmoid":
            continue
        if name == "theta_temp" and config.temperature == "none":
            continue
        row = {"parameter": name, "estimate": value, "se": se,
               "ci_low": value - 1.96 * se, "ci_high": value + 1.96 * se}
        if name.startswith("theta_") and name != "theta_temp":
            row["relative_change_in_R_percent"] = 100 * (np.exp(value) - 1)
            row["ci_low_percent"] = 100 * (np.exp(value - 1.96 * se) - 1)
            row["ci_high_percent"] = 100 * (np.exp(value + 1.96 * se) - 1)
        elif name == "theta_temp" and config.temperature == "linear":
            row["relative_change_in_R_percent"] = 100 * (np.exp(-value) - 1)  # per 1 °C colder
            row["ci_low_percent"] = 100 * (np.exp(-(value + 1.96 * se)) - 1)
            row["ci_high_percent"] = 100 * (np.exp(-(value - 1.96 * se)) - 1)
        elif name == "gamma_red":  # fraction of reports postponed on weekday public holidays
            row["postponed_share"] = 1 / (1 + np.exp(-value))
        rows.append(row)
    # R at a range of temperatures relative to 5 °C, with 95 % intervals from
    # draws of the Laplace approximation.
    if config.temperature != "none":
        rng = np.random.default_rng(7)
        draws = rng.multivariate_normal(effects["theta"], effects["cov"], size=4000)
        curves = np.array([temperature_multiplier(d, TEMPERATURE_GRID, config.temperature) for d in draws])
        point = temperature_multiplier(effects["theta"], TEMPERATURE_GRID, config.temperature)
        for i, temp in enumerate(TEMPERATURE_GRID):
            rows.append({
                "parameter": f"R_at_{temp:g}C_vs_5C", "estimate": point[i], "se": np.nan,
                "relative_change_in_R_percent": 100 * (point[i] - 1),
                "ci_low_percent": 100 * (np.quantile(curves[:, i], 0.025) - 1),
                "ci_high_percent": 100 * (np.quantile(curves[:, i], 0.975) - 1),
            })
    return pd.DataFrame(rows)


def prior_path(variant: str) -> Path:
    return RESULTS / f"forecast_prior_{variant}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="main")
    parser.add_argument("--skip-loso", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    wide = load_history()
    config = variant_config(args.variant)
    suffix = "" if args.variant == "main" else f"_{args.variant}"
    print("Generation interval:", generation_summary(config.delays))
    effects = estimate_effects(config, wide=wide, verbose=True)
    print(f"log evidence {effects['log_evidence']:.2f}, |g| {effects['grad_max']:.1e}")
    table = effects_table(effects, config)
    print(table.to_string(index=False))
    table.to_csv(RESULTS / f"effects{suffix}.csv", index=False)
    fitted_table(effects).to_csv(RESULTS / f"fitted_training_seasons{suffix}.csv", index=False)
    prior, k_table = series_priors(config, effects, wide=wide)
    k_table.to_csv(RESULTS / f"susceptible_pool_by_season{suffix}.csv", index=False)
    print(k_table.to_string(index=False))
    with prior_path(args.variant).open("w") as handle:
        json.dump({"variant": args.variant, "config": repr(config), **prior.to_dict(),
                   "log_evidence": effects["log_evidence"], "loglik": effects["loglik"],
                   "names": shared_names()}, handle, indent=1)
    if not args.skip_loso:
        loso = leave_one_season_out(config, wide=wide)
        loso.to_csv(RESULTS / f"leave_one_season_out{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
