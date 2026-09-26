"""Model configurations: the main model and the alternatives used for comparison.

The model is an ODE SEIR with exponentially distributed latent (mean 1.6
days) and infectious (mean 2.6 days) periods, solved with RK4 at two steps
per day. The infection-to-report delay (gamma, mean 11 days, SD 4.5) and the
random-walk scale (0.07 per week) were chosen on the training seasons
(2015/16 to 2024/25 without the COVID-19 seasons) by Laplace-approximated
marginal likelihood; see ``seirflu.selection``.

The main model uses a log-linear temperature effect. A sigmoid effect that
levels off at cold and warm temperatures fits the training seasons better
(10.6 log units), but the log-linear effect gave lower forecast errors in all
three backtest seasons (2022/23 to 2024/25) and in all three locations, so it
is used for the submitted forecasts; ``sigmoid-temperature`` keeps the
alternative.
"""

from __future__ import annotations

from dataclasses import replace

from .model import HOLIDAY_TYPES, DelayConfig, ModelConfig

MAIN_DELAYS = DelayConfig(latent_mean=1.6, infectious_mean=2.6, report_mean=11.0, report_sd=4.5)
MAIN_CONFIG = ModelConfig(sigma_rw=0.07, delays=MAIN_DELAYS, temperature="linear")

VARIANTS: dict[str, ModelConfig] = {
    "main": MAIN_CONFIG,
    "sigmoid-temperature": MAIN_CONFIG.with_(temperature="sigmoid"),
    "no-temperature": MAIN_CONFIG.with_(temperature="none"),
    "no-holidays": MAIN_CONFIG.with_(holidays=()),
    "no-holidays-no-temperature": MAIN_CONFIG.with_(holidays=(), temperature="none"),
}


def with_delay(config: ModelConfig, **changes) -> ModelConfig:
    return config.with_(delays=replace(config.delays, **changes))


def without_summer(config: ModelConfig) -> ModelConfig:
    return config.with_(holidays=tuple(h for h in HOLIDAY_TYPES if h != "sommar"))


def variant_config(name: str) -> ModelConfig:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown variant {name!r}; choose from {sorted(VARIANTS)}") from exc
