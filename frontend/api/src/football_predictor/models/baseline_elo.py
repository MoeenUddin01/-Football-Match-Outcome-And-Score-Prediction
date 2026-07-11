"""Elo-based baseline benchmark model for match outcome prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config(config: Any | None = None) -> dict[str, Any]:
    """Load config from a dict or the repository YAML file."""
    if config is None:
        config_path = Path(__file__).resolve().parents[3] / "config" / "config.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            loaded_config = yaml.safe_load(handle) or {}
        return loaded_config

    if isinstance(config, dict):
        return config

    if isinstance(config, (str, Path)):
        with Path(config).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    raise TypeError("config must be a dict, YAML path, or None")


def compute_draw_rate(outcomes: pd.Series) -> float:
    """Compute the empirical draw rate from a series of outcomes.

    Parameters
    ----------
    outcomes : pd.Series
        Series of outcome labels ('home_win', 'draw', 'away_win').

    Returns
    -------
    float
        Proportion of draws in the series.
    """
    total = len(outcomes)
    if total == 0:
        return 0.0
    return (outcomes == "draw").sum() / total


def compute_elo_expected_score(
    elo_diff: float,
    neutral: bool,
    home_advantage: float = 100.0,
) -> float:
    """Compute expected home score from Elo rating difference.

    Parameters
    ----------
    elo_diff : float
        Home rating minus away rating (before home advantage adjustment).
    neutral : bool
        Whether the match is at a neutral venue.
    home_advantage : float
        Home advantage constant (default 100).

    Returns
    -------
    float
        Expected score for the home team (We_home).
    """
    if neutral:
        rating_difference = elo_diff
    else:
        rating_difference = elo_diff + home_advantage

    return 1.0 / (10.0 ** (-rating_difference / 400.0) + 1.0)


def predict_elo_baseline(
    elo_diff: float | pd.Series,
    neutral: bool | pd.Series,
    draw_rate: float,
    config: Any | None = None,
) -> dict[str, float] | pd.DataFrame:
    """Predict match outcome probabilities using Elo baseline.

    Parameters
    ----------
    elo_diff : float or pd.Series
        Home Elo rating minus away Elo rating.
    neutral : bool or pd.Series
        Whether the match is at a neutral venue.
    draw_rate : float
        Empirical draw rate from the training set.
    config : dict, optional
        Configuration dictionary. If None, loads from config.yaml.

    Returns
    -------
    dict or pd.DataFrame
        If scalar inputs: dict with keys 'home_win_prob', 'draw_prob', 'away_win_prob'.
        If Series inputs: DataFrame with those three columns.
    """
    config_data = _load_config(config)
    elo_params = config_data.get("elo_params", {})
    home_advantage = float(elo_params.get("home_advantage", 100.0))

    if isinstance(elo_diff, (int, float)) and isinstance(neutral, bool):
        we_home = compute_elo_expected_score(float(elo_diff), neutral, home_advantage)
        we_away = 1.0 - we_home

        p_home_win = (1.0 - draw_rate) * (we_home / (we_home + we_away))
        p_away_win = (1.0 - draw_rate) * (we_away / (we_home + we_away))

        return {
            "home_win_prob": p_home_win,
            "draw_prob": draw_rate,
            "away_win_prob": p_away_win,
        }

    elo_diff_series = pd.Series(elo_diff) if isinstance(elo_diff, (list, np.ndarray)) else elo_diff
    neutral_series = pd.Series(neutral) if isinstance(neutral, (list, np.ndarray)) else neutral

    we_home = pd.Series(index=elo_diff_series.index, dtype=float)
    for idx in elo_diff_series.index:
        ed = float(elo_diff_series[idx])
        n = bool(neutral_series[idx]) if isinstance(neutral_series, pd.Series) else bool(neutral_series)
        we_home[idx] = compute_elo_expected_score(ed, n, home_advantage)
    we_away = 1.0 - we_home

    p_home_win = (1.0 - draw_rate) * (we_home / (we_home + we_away))
    p_away_win = (1.0 - draw_rate) * (we_away / (we_home + we_away))
    p_draw = pd.Series(draw_rate, index=elo_diff_series.index)

    return pd.DataFrame({
        "home_win_prob": p_home_win,
        "draw_prob": p_draw,
        "away_win_prob": p_away_win,
    })
