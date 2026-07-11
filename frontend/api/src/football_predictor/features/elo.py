"""Chronological Elo rating engine for match outcome prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_WARNED_UNMAPPED_TOURNAMENTS: set[str] = set()


def _load_config(config: Any | None = None) -> dict[str, Any]:
    """Load Elo config from a dict or the repository YAML file."""
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


def _normalize_tournament_name(tournament: str | None) -> str:
    """Normalize a tournament string for tier lookup."""
    if tournament is None:
        return ""
    return str(tournament).strip().lower()


def _resolve_tier_name(tournament: str | None) -> str:
    """Resolve a tournament tier from the explicit lookup table."""
    normalized = _normalize_tournament_name(tournament)
    lookup_table = {
        "friendly": "friendly",
        "minor tournament": "minor_tournament",
        "minor_tournament": "minor_tournament",
        "fifa world cup": "world_cup",
        "uefa euro": "continental_championship",
        "copa america": "continental_championship",
        "copa américa": "continental_championship",
        "african cup of nations": "continental_championship",
        "afc asian cup": "continental_championship",
        "gold cup": "continental_championship",
    }

    if normalized in lookup_table:
        return lookup_table[normalized]
    if "qualification" in normalized:
        return "qualifier"
    return "default"


def get_k_factor(tournament: str, config: Any | None = None) -> int:
    """Return the K-factor for a tournament using the configured tier map."""
    config_data = _load_config(config)
    k_factor_tiers = config_data.get("elo_params", {}).get("k_factor_tiers", {})
    tier_name = _resolve_tier_name(tournament)

    if tier_name == "default" and tournament not in _WARNED_UNMAPPED_TOURNAMENTS:
        logger.warning(
            "Tournament '%s' fell back to default K-factor; extend the mapping if needed.",
            tournament,
        )
        _WARNED_UNMAPPED_TOURNAMENTS.add(str(tournament))

    return int(k_factor_tiers.get(tier_name, k_factor_tiers.get("default", 20)))


def resolve_k_factor(tournament: str, config: Any | None = None) -> int:
    """Backward-compatible alias for get_k_factor."""
    return get_k_factor(tournament, config=config)


def get_goal_difference_multiplier(goal_difference: int | float) -> float:
    """Return the Elo goal-difference multiplier for a given margin."""
    absolute_margin = abs(int(goal_difference))
    if absolute_margin <= 1:
        return 1.0
    if absolute_margin == 2:
        return 1.5
    return (11 + absolute_margin) / 8.0


def calculate_elo_update(
    home_rating: float,
    away_rating: float,
    home_score: int | float,
    away_score: int | float,
    tournament: str,
    neutral: bool,
    home_advantage: float,
    initial_rating: float,
    k_factor: float,
) -> tuple[float, float, float, float]:
    """Return updated Elo ratings and expected scores for a single match."""
    actual_score = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
    goal_difference = abs(int(home_score) - int(away_score))
    multiplier = get_goal_difference_multiplier(goal_difference)

    if neutral:
        rating_difference = home_rating - away_rating
    else:
        rating_difference = (home_rating + home_advantage) - away_rating

    expected_home = 1 / (10 ** (-rating_difference / 400.0) + 1.0)
    expected_away = 1.0 - expected_home

    home_new = home_rating + k_factor * multiplier * (actual_score - expected_home)
    away_new = away_rating + k_factor * multiplier * ((1.0 - actual_score) - expected_away)

    return home_new, away_new, expected_home, expected_away


def compute_elo_ratings(results_df: pd.DataFrame, config: Any | None = None) -> pd.DataFrame:
    """Compute chronological Elo ratings and return the input frame with pre/post columns."""
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame")

    if "date" not in results_df.columns:
        raise KeyError("results_df must contain a 'date' column")

    dates = pd.to_datetime(results_df["date"])
    if not dates.is_monotonic_increasing:
        raise ValueError("results_df must be sorted by date in ascending order")

    config_data = _load_config(config)
    elo_params = config_data.get("elo_params", {})
    initial_rating = float(elo_params.get("initial_rating", 1500.0))
    home_advantage = float(elo_params.get("home_advantage", 100.0))

    ratings: dict[str, float] = {}
    elo_results = results_df.copy()
    elo_results["home_elo_pre"] = pd.NA
    elo_results["away_elo_pre"] = pd.NA
    elo_results["home_elo_post"] = pd.NA
    elo_results["away_elo_post"] = pd.NA

    for idx, row in elo_results.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        home_pre = ratings.get(home_team, initial_rating)
        away_pre = ratings.get(away_team, initial_rating)

        elo_results.at[idx, "home_elo_pre"] = home_pre
        elo_results.at[idx, "away_elo_pre"] = away_pre

        k_factor = get_k_factor(str(row.get("tournament", "")), config=config_data)
        home_new, away_new, _, _ = calculate_elo_update(
            home_rating=home_pre,
            away_rating=away_pre,
            home_score=row.get("home_score", 0),
            away_score=row.get("away_score", 0),
            tournament=str(row.get("tournament", "")),
            neutral=bool(row.get("neutral", False)),
            home_advantage=home_advantage,
            initial_rating=initial_rating,
            k_factor=k_factor,
        )

        ratings[home_team] = home_new
        ratings[away_team] = away_new

        elo_results.at[idx, "home_elo_post"] = home_new
        elo_results.at[idx, "away_elo_post"] = away_new

    return elo_results


def get_rating_history(team_name: str, elo_results_df: pd.DataFrame) -> pd.DataFrame:
    """Return a team's rating over time from the Elo results frame."""
    if not isinstance(elo_results_df, pd.DataFrame):
        raise TypeError("elo_results_df must be a pandas DataFrame")

    if {"date", "home_team", "away_team", "home_elo_post", "away_elo_post"} - set(elo_results_df.columns):
        raise KeyError("elo_results_df must contain Elo output columns")

    history_rows: list[dict[str, Any]] = []
    for _, row in elo_results_df.iterrows():
        if str(row["home_team"]) == team_name:
            history_rows.append({"date": row["date"], "rating": row["home_elo_post"]})
        elif str(row["away_team"]) == team_name:
            history_rows.append({"date": row["date"], "rating": row["away_elo_post"]})

    history_df = pd.DataFrame(history_rows, columns=["date", "rating"])
    if history_df.empty:
        return history_df

    history_df["date"] = pd.to_datetime(history_df["date"])
    return history_df.sort_values("date").reset_index(drop=True)
