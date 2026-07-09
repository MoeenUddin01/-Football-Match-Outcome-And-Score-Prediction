"""Tests for the compute_elo_artifacts script."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.compute_elo_artifacts import (
    build_tournament_tier_map,
    extract_latest_elo_ratings,
)
from src.football_predictor.features.elo import compute_elo_ratings
from src.football_predictor.utils.io import load_artifact


@pytest.fixture
def sample_results_df() -> pd.DataFrame:
    """Create a minimal results DataFrame for testing."""
    data = {
        "date": [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-05",
        ],
        "home_team": ["Brazil", "Argentina", "Germany", "France", "Spain"],
        "away_team": ["Argentina", "Germany", "France", "Spain", "Brazil"],
        "home_score": [2, 1, 0, 3, 1],
        "away_score": [1, 1, 2, 1, 2],
        "tournament": [
            "FIFA World Cup",
            "Friendly",
            "UEFA Euro",
            "FIFA World Cup qualification",
            "Copa America",
        ],
        "neutral": [False, True, False, False, True],
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_extract_latest_elo_ratings(sample_results_df: pd.DataFrame) -> None:
    """Test that extract_latest_elo_ratings extracts the most recent rating."""
    elo_results = compute_elo_ratings(sample_results_df)
    latest_ratings = extract_latest_elo_ratings(elo_results)

    # Check that all teams are present
    expected_teams = {"Brazil", "Argentina", "Germany", "France", "Spain"}
    assert set(latest_ratings.keys()) == expected_teams

    # Check that ratings are floats
    for rating in latest_ratings.values():
        assert isinstance(rating, float)


def test_build_tournament_tier_map() -> None:
    """Test that build_tournament_tier_map returns expected mappings."""
    tier_map = build_tournament_tier_map()

    # Check that expected tournaments are present
    assert "FIFA World Cup" in tier_map
    assert "Friendly" in tier_map
    assert "UEFA Euro" in tier_map
    assert "Copa America" in tier_map

    # Check tier assignments
    assert tier_map["FIFA World Cup"] == "world_cup"
    assert tier_map["Friendly"] == "friendly"
    assert tier_map["UEFA Euro"] == "continental_championship"
    assert tier_map["Copa America"] == "continental_championship"


def test_artifacts_can_be_loaded(tmp_path: Path, sample_results_df: pd.DataFrame) -> None:
    """Test that artifacts can be saved and loaded."""
    from src.football_predictor.utils.io import save_artifact

    # Create and save team_elo_latest.json
    elo_results = compute_elo_ratings(sample_results_df)
    latest_ratings = extract_latest_elo_ratings(elo_results)
    save_artifact(latest_ratings, "team_elo_latest.json", tmp_path)

    # Create and save tournament_tier_map.json
    tier_map = build_tournament_tier_map()
    save_artifact(tier_map, "tournament_tier_map.json", tmp_path)

    # Load and verify
    loaded_ratings = load_artifact("team_elo_latest.json", tmp_path)
    loaded_tier_map = load_artifact("tournament_tier_map.json", tmp_path)

    assert loaded_ratings == latest_ratings
    assert loaded_tier_map == tier_map
