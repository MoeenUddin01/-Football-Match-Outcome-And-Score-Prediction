"""Chronological rolling statistics engine for match outcome prediction."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config(config: Any | None = None) -> dict[str, Any]:
    """Load rolling stats config from a dict or the repository YAML file."""
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


def _normalize_window_sizes(raw_params: dict[str, Any]) -> dict[str, list[int]]:
    """Normalize window sizes from config to lists of ints.

    Handles both single int values and lists from config.
    """
    normalized: dict[str, list[int]] = {}
    for key, value in raw_params.items():
        if isinstance(value, list):
            normalized[key] = [int(v) for v in value]
        else:
            normalized[key] = [int(value)]
    return normalized


def compute_rolling_features(
    results_df: pd.DataFrame,
    window_sizes: dict[str, int | list[int]] | None = None,
    config: Any | None = None,
) -> pd.DataFrame:
    """Compute rolling features for each match using only prior matches.

    Processes matches in chronological order in a single forward pass.
    For match N, only uses information from matches 1..N-1 to prevent
    temporal leakage.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: date, home_team, away_team, home_score, away_score.
        Must be sorted by date in ascending order.
    window_sizes : dict, optional
        Dictionary with keys 'form_window', 'goals_window', 'h2h_window'.
        Values can be single ints or lists of ints. If None, loads from config.
    config : dict, optional
        Configuration dictionary. If None, loads from config.yaml.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with rolling feature columns.
    """
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame")

    required_cols = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing_cols = required_cols - set(results_df.columns)
    if missing_cols:
        raise KeyError(f"results_df must contain columns: {missing_cols}")

    dates = pd.to_datetime(results_df["date"])
    if not dates.is_monotonic_increasing:
        raise ValueError("results_df must be sorted by date in ascending order")

    config_data = _load_config(config)
    rolling_params = config_data.get("rolling_window_sizes", {})

    if window_sizes is None:
        window_sizes = _normalize_window_sizes(rolling_params)
    else:
        normalized: dict[str, list[int]] = {}
        for key, value in window_sizes.items():
            if isinstance(value, list):
                normalized[key] = [int(v) for v in value]
            else:
                normalized[key] = [int(value)]
        window_sizes = normalized

    goals_windows = window_sizes.get("goals_window", [5])
    form_windows = window_sizes.get("form_window", [5])
    h2h_windows = window_sizes.get("h2h_window", [5])

    max_window = max(
        max(goals_windows),
        max(form_windows),
        max(h2h_windows),
    )

    output = results_df.copy()
    output["date"] = pd.to_datetime(output["date"])

    team_history: dict[str, deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=max_window * 2)
    )
    h2h_history: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=max(h2h_windows) * 2)
    )

    FIRST_MATCH_DEFAULT_REST_DAYS = 30

    for gw in goals_windows:
        output[f"home_goals_scored_avg_last_{gw}"] = 0.0
        output[f"home_goals_conceded_avg_last_{gw}"] = 0.0
        output[f"away_goals_scored_avg_last_{gw}"] = 0.0
        output[f"away_goals_conceded_avg_last_{gw}"] = 0.0

    for fw in form_windows:
        output[f"home_win_rate_last_{fw}"] = 0.0
        output[f"away_win_rate_last_{fw}"] = 0.0

    output["home_rest_days"] = FIRST_MATCH_DEFAULT_REST_DAYS
    output["away_rest_days"] = FIRST_MATCH_DEFAULT_REST_DAYS
    output["h2h_home_wins"] = 0
    output["h2h_away_wins"] = 0
    output["h2h_draws"] = 0

    for idx, row in output.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        match_date = row["date"]

        home_score = int(row.get("home_score", 0))
        away_score = int(row.get("away_score", 0))

        home_history = list(team_history[home_team])
        away_history = list(team_history[away_team])

        if len(home_history) >= 1:
            last_home_match = home_history[-1]
            rest_days = (match_date - last_home_match["date"]).days
            output.at[idx, "home_rest_days"] = rest_days
        else:
            output.at[idx, "home_rest_days"] = FIRST_MATCH_DEFAULT_REST_DAYS

        if len(away_history) >= 1:
            last_away_match = away_history[-1]
            rest_days = (match_date - last_away_match["date"]).days
            output.at[idx, "away_rest_days"] = rest_days
        else:
            output.at[idx, "away_rest_days"] = FIRST_MATCH_DEFAULT_REST_DAYS

        for gw in goals_windows:
            if len(home_history) >= 1:
                recent_home = home_history[-gw:]
                output.at[idx, f"home_goals_scored_avg_last_{gw}"] = sum(
                    m["goals_scored"] for m in recent_home
                ) / len(recent_home)
                output.at[idx, f"home_goals_conceded_avg_last_{gw}"] = sum(
                    m["goals_conceded"] for m in recent_home
                ) / len(recent_home)

            if len(away_history) >= 1:
                recent_away = away_history[-gw:]
                output.at[idx, f"away_goals_scored_avg_last_{gw}"] = sum(
                    m["goals_scored"] for m in recent_away
                ) / len(recent_away)
                output.at[idx, f"away_goals_conceded_avg_last_{gw}"] = sum(
                    m["goals_conceded"] for m in recent_away
                ) / len(recent_away)

        for fw in form_windows:
            if len(home_history) >= 1:
                recent_form_home = home_history[-fw:]
                output.at[idx, f"home_win_rate_last_{fw}"] = sum(
                    1 for m in recent_form_home if m["result"] == "win"
                ) / len(recent_form_home)

            if len(away_history) >= 1:
                recent_form_away = away_history[-fw:]
                output.at[idx, f"away_win_rate_last_{fw}"] = sum(
                    1 for m in recent_form_away if m["result"] == "win"
                ) / len(recent_form_away)

        h2h_key = (home_team, away_team) if home_team < away_team else (away_team, home_team)
        h2h_matches = list(h2h_history[h2h_key])
        if len(h2h_matches) >= 1:
            recent_h2h = h2h_matches[-max(h2h_windows):]
            for h2h_match in recent_h2h:
                if h2h_match["home_team"] == home_team:
                    if h2h_match["home_score"] > h2h_match["away_score"]:
                        output.at[idx, "h2h_home_wins"] += 1
                    elif h2h_match["home_score"] < h2h_match["away_score"]:
                        output.at[idx, "h2h_away_wins"] += 1
                    else:
                        output.at[idx, "h2h_draws"] += 1
                else:
                    if h2h_match["home_score"] > h2h_match["away_score"]:
                        output.at[idx, "h2h_away_wins"] += 1
                    elif h2h_match["home_score"] < h2h_match["away_score"]:
                        output.at[idx, "h2h_home_wins"] += 1
                    else:
                        output.at[idx, "h2h_draws"] += 1

        if home_score > away_score:
            home_result = "win"
            away_result = "loss"
        elif home_score < away_score:
            home_result = "loss"
            away_result = "win"
        else:
            home_result = "draw"
            away_result = "draw"

        team_history[home_team].append({
            "date": match_date,
            "goals_scored": home_score,
            "goals_conceded": away_score,
            "result": home_result,
        })
        team_history[away_team].append({
            "date": match_date,
            "goals_scored": away_score,
            "goals_conceded": home_score,
            "result": away_result,
        })

        h2h_history[h2h_key].append({
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
        })

    return output
