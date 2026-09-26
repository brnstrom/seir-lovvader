"""Forecasts for one hub round: fit each location to data up to the cutoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .data import HUB_LOCATIONS, season_of, window_start
from .fit import Problem, SharedPrior
from .model import W_MAX, ModelConfig
from .units import build_unit

HORIZONS = (0, 1, 2, 3)
QUANTILES = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)


@dataclass
class ForecastSettings:
    config: ModelConfig
    prior: SharedPrior
    weather_mode: str = "climatology"  # or "oracle" (uses observed future weather)
    n_samples: int = 1000
    seed: int = 20251005
    # Scale of the random-walk deviation of log R after the data cutoff:
    # 1 = continue the fitted random walk, 0 = keep R at its cutoff value
    # (holiday and temperature effects still apply). The first value is the
    # one written to hub files.
    future_rw_scales: tuple[float, ...] = (1.0, 0.5, 0.0)
    locations: tuple[str, ...] = HUB_LOCATIONS
    warm_start: dict = field(default_factory=dict)


def _nb_draws(mu: np.ndarray, phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shape = np.broadcast_to(phi[:, None], mu.shape)
    rate = rng.gamma(shape, np.clip(mu, 1e-9, None) / shape)
    return rng.poisson(rate)


def forecast_location(
    series: str,
    reference_date: date,
    weekly: pd.Series,
    settings: ForecastSettings,
) -> tuple[list[dict], dict]:
    cutoff = reference_date - timedelta(days=7)
    season = season_of(cutoff)
    window_end = reference_date + timedelta(days=21)
    start = window_start(season)
    if (window_end - start).days + 1 > 7 * W_MAX:
        raise ValueError("Forecast window exceeds the model window")
    unit = build_unit(
        series, season, weekly, settings.config,
        last_observed=cutoff, window_end=window_end,
        weather_cutoff=cutoff, weather_mode=settings.weather_mode,
        log_k_prior=settings.prior.log_k[series],
    )
    problem = Problem([unit], settings.config, shared_prior=settings.prior, series_names=[series])
    warm_key = (series, season)
    flat0 = settings.warm_start.get(warm_key)
    result = None
    if flat0 is not None:
        try:
            result = problem.fit(flat0=flat0)
        except FloatingPointError:
            result = None
    # Also fit from the default start, which guards against a poor warm start.
    fresh = problem.fit()
    if result is None or fresh.fun < result.fun:
        result = fresh
    settings.warm_start[warm_key] = result.x
    _, chol, _ = problem.laplace(result.x)
    base_samples = problem.sample(result.x, chol, settings.n_samples, seed=settings.seed)
    phi = np.exp(np.array([problem.params(s)["shared"]["log_phi"][0] for s in base_samples]))
    knot_idx = problem.log_r_indices(0)
    cutoff_day = (cutoff - start).days
    # Knots sit on Mondays; the cutoff is a Sunday, so knot ``anchor`` is the
    # first one after the cutoff and knots beyond it only affect the future.
    anchor = (cutoff_day + 1) // 7

    map_weekly, _, map_r, map_reff = problem.simulate_flat(result.x)
    map_weekly = np.asarray(map_weekly)[0]
    params = problem.params(result.x)
    rows = []
    r_eff = None
    for scale in settings.future_rw_scales:
        samples = base_samples.copy()
        knots = samples[:, knot_idx]
        future = knots[:, anchor + 1:]
        samples[:, knot_idx[anchor + 1:]] = knots[:, [anchor]] + scale * (future - knots[:, [anchor]])
        weekly_mu, r_eff_s = problem.simulate_samples(samples)
        weekly_mu = weekly_mu[:, 0, :]
        if r_eff is None:
            r_eff = r_eff_s
        rng = np.random.default_rng(settings.seed + 7)
        draws = _nb_draws(weekly_mu, phi, rng)
        for horizon in HORIZONS:
            target_end = reference_date + timedelta(days=7 * horizon)
            w = ((target_end - start).days + 1) // 7 - 1
            mu_w = weekly_mu[:, w]
            row = {
                "reference_date": reference_date.isoformat(),
                "location": series,
                "horizon": horizon,
                "target_end_date": target_end.isoformat(),
                "future_rw_scale": scale,
                "mean": float(np.mean(mu_w)),
                "map_mean": float(map_weekly[w]),
            }
            for q in QUANTILES:
                row[f"q{q:g}"] = float(np.quantile(draws[:, w], q))
            rows.append(row)
    diagnostics = {
        "reference_date": reference_date.isoformat(),
        "location": series,
        "r_eff_cutoff": float(np.asarray(map_reff)[0, cutoff_day]),
        "r_eff_cutoff_q10": float(np.quantile(r_eff[:, 0, cutoff_day], 0.1)),
        "r_eff_cutoff_q90": float(np.quantile(r_eff[:, 0, cutoff_day], 0.9)),
        "log_k": float(params["units"]["log_k"][0]),
        "log_k_prior_mean": float(unit.log_k_prior[0]),
        "susceptible_fraction_cutoff": float(np.asarray(map_reff)[0, cutoff_day] / np.asarray(map_r)[0, cutoff_day]),
        "phi": float(np.exp(params["shared"]["log_phi"][0])),
        "objective": float(result.fun),
        "grad_max": float(np.max(np.abs(result.jac))),
    }
    for name, value in zip(("host", "jul", "sport", "pask", "sommar"), params["shared"]["theta_hol"]):
        diagnostics[f"theta_{name}"] = float(value)
    for key in ("theta_temp", "temp_mid", "temp_log_width"):
        diagnostics[key] = float(params["shared"][key])
    return rows, diagnostics


def forecast_round(reference_date: date, wide: pd.DataFrame, settings: ForecastSettings):
    rows, diagnostics = [], []
    for series in settings.locations:
        r, d = forecast_location(series, reference_date, wide[series], settings)
        rows.extend(r)
        diagnostics.append(d)
    return rows, diagnostics
