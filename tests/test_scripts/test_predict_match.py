"""Tests for the predict_match script."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.predict_match import (
    _get_team_elo,
    _get_team_form,
    _resolve_tier,
    _load_artifacts,
    predict_match,
)


@pytest.fixture(scope="module")
def artifacts() -> dict:
    return _load_artifacts()


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    return pd.read_parquet("data/processed/features.parquet")


def test_known_matchup_runs(artifacts: dict, features_df: pd.DataFrame) -> None:
    """Confirm a real known matchup runs end-to-end without error."""
    result = predict_match("Brazil", "Argentina", "FIFA World Cup", True, artifacts, features_df)
    assert result is not None
    assert "home_win" in result["xgb_probs"]
    assert "draw" in result["xgb_probs"]
    assert "away_win" in result["xgb_probs"]
    assert result["poisson_home_goals"] >= 0
    assert result["poisson_away_goals"] >= 0


def test_unknown_team_handled(artifacts: dict, features_df: pd.DataFrame) -> None:
    """Confirm an unknown team returns None, not a crash."""
    result = predict_match("Narnia", "Wakanda", "Friendly", False, artifacts, features_df)
    assert result is None


def test_get_team_elo_known() -> None:
    """Test Elo lookup for a known team."""
    elo = {"Brazil": 2050.0, "Argentina": 1980.0}
    assert _get_team_elo("Brazil", elo) == 2050.0
    assert _get_team_elo("Unknown", elo) is None


def test_resolve_tier() -> None:
    """Test tournament tier resolution."""
    tmap = {"FIFA World Cup": "world_cup", "Friendly": "friendly"}
    assert _resolve_tier("FIFA World Cup", tmap) == "world_cup"
    assert _resolve_tier("Unknown Cup", tmap) == "default"


def test_get_team_form(artifacts: dict, features_df: pd.DataFrame) -> None:
    """Test that form lookup returns expected keys for a known team."""
    form = _get_team_form("Brazil", "home", features_df)
    assert "home_goals_scored_avg_last_5" in form
    assert "home_win_rate_last_5" in form
