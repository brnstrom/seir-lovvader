"""Construction of model units (one series in one fitting window)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .calendar import daily_holiday_exposure, red_days
from .data import season_end, season_start, window_start
from .model import HOLIDAY_TYPES, W_MAX, ModelConfig, Unit, initial_log_r
from .weather import forecast_weather, series_weather

# Seasons whose dynamics were dominated by COVID-19 measures.
EXCLUDED_SEASONS = (2020, 2021)
# The 2019/20 season is truncated when COVID-19 measures began in Sweden.
TRUNCATE = {2019: date(2020, 3, 15)}


def window_days(start: date) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=7 * W_MAX, freq="D")


def iota_prior(y: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    observed = y[obs > 0][:6]
    base = float(np.mean(observed)) if len(observed) else 5.0
    return float(np.log(max(base, 0.5) / 7.0 * 0.3)), 1.5


def build_unit(
    series: str,
    season: int,
    weekly: pd.Series,
    config: ModelConfig,
    *,
    last_observed: date | None = None,
    window_end: date | None = None,
    weather_cutoff: date | None = None,
    weather_mode: str = "climatology",
    log_k_prior: tuple[float, float] | None = None,
) -> Unit:
    """Build a unit for ``series`` in ``season``.

    ``weekly`` is indexed by week-ending Sunday. Only weeks up to
    ``last_observed`` enter the likelihood. Weather after ``weather_cutoff``
    is replaced by the forecast (climatology + decaying anomaly) unless
    ``weather_mode='oracle'``; with ``weather_cutoff=None`` observed weather is
    used throughout (historical estimation).
    """
    start = window_start(season)
    first_observed = season_start(season)
    end = window_end or season_end(season)
    n_weeks = ((end - start).days + 1) // 7
    if n_weeks > W_MAX:
        raise ValueError("Window longer than W_MAX weeks")
    days = window_days(start)
    week_ends = [start + timedelta(days=7 * (w + 1) - 1) for w in range(W_MAX)]
    limit = min(d for d in (last_observed, TRUNCATE.get(season), end) if d is not None)
    y = np.zeros(W_MAX)
    obs = np.zeros(W_MAX)
    for w, week_end in enumerate(week_ends):
        if w >= n_weeks or week_end > limit or week_end < first_observed:
            continue
        value = weekly.get(pd.Timestamp(week_end), np.nan)
        if np.isfinite(value):
            y[w] = value
            obs[w] = 1.0

    hol = daily_holiday_exposure(series).reindex(days).fillna(0.0)[list(HOLIDAY_TYPES)].to_numpy()
    red = red_days().reindex(days).fillna(0.0).to_numpy()
    red_dest = np.arange(len(days))
    weekday = days.weekday.to_numpy()
    for i in np.where(red > 0)[0]:
        j = i + 1
        while j < len(days) and (red[j] > 0 or weekday[j] >= 5):
            j += 1
        red_dest[i] = min(j, len(days) - 1)
    if weather_cutoff is None:
        temp = series_weather(series)["temperature"].reindex(days).interpolate(limit_direction="both").to_numpy()
    else:
        temp = forecast_weather(series, "temperature", days, pd.Timestamp(weather_cutoff), mode=weather_mode)

    if log_k_prior is None:
        total = float(y[obs > 0].sum())
        log_k_prior = (float(np.log(3.0 * total + 300.0)), 1.5)
    return Unit(
        series=series,
        season=season,
        start=start,
        n_weeks=n_weeks,
        y=y,
        obs=obs,
        hol=hol,
        temp=temp,
        red=red,
        red_dest=red_dest,
        log_k_prior=log_k_prior,
        log_iota_prior=iota_prior(y, obs),
        init_log_r=initial_log_r(y, obs, config.delays),
    )


# Historical windows end in ISO week 23 (early June). Forecast windows never
# extend beyond week 23 either, and the end-of-season tail in June-July is
# dominated by strain replacement and sporadic cases.
HISTORICAL_END_WEEK = 23


def historical_units(
    wide: pd.DataFrame,
    series_list: tuple[str, ...],
    seasons: tuple[int, ...],
    config: ModelConfig,
    end_week: int | None = HISTORICAL_END_WEEK,
) -> list[Unit]:
    units = []
    for season in seasons:
        if season in EXCLUDED_SEASONS:
            continue
        window_end = date.fromisocalendar(season + 1, end_week, 7) if end_week else None
        for series in series_list:
            units.append(build_unit(series, season, wide[series], config, window_end=window_end))
    return units
