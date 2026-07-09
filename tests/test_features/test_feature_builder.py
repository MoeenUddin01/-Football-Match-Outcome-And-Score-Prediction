"""Tests for the feature builder module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from football_predictor.features.elo import compute_elo_ratings
from football_predictor.features.feature_builder import build_feature_table, save_feature_table
from football_predictor.utils.io import load_artifact


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
    return pd.DataFrame(data)


def test_row_count_matches_input(sample_results_df: pd.DataFrame) -> None:
    """Test that output row count matches input row count."""
    feature_df = build_feature_table(sample_results_df)
    assert len(feature_df) == len(sample_results_df)


def test_elo_diff_calculation(sample_results_df: pd.DataFrame) -> None:
    """Test that elo_diff is calculated correctly."""
    feature_df = build_feature_table(sample_results_df)

    for idx in range(len(feature_df)):
        expected = feature_df.iloc[idx]["home_elo_pre"] - feature_df.iloc[idx]["away_elo_pre"]
        actual = feature_df.iloc[idx]["elo_diff"]
        assert actual == pytest.approx(expected, abs=0.01), (
            f"Row {idx}: elo_diff={actual}, expected={expected}"
        )


def test_outcome_reflects_actual_score(sample_results_df: pd.DataFrame) -> None:
    """Test that outcome correctly reflects actual match result."""
    feature_df = build_feature_table(sample_results_df)

    # Match 0: Brazil 2-1 Argentina -> home_win
    assert feature_df.iloc[0]["outcome"] == "home_win"

    # Match 1: Argentina 1-1 Germany -> draw
    assert feature_df.iloc[1]["outcome"] == "draw"

    # Match 2: Germany 0-2 France -> away_win
    assert feature_df.iloc[2]["outcome"] == "away_win"


def test_goal_diff_and_outcome_not_in_feature_columns(
    sample_results_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test that goal_diff and outcome are not in the saved feature_columns.json."""
    feature_df = build_feature_table(sample_results_df)

    # Mock config to use tmp_path
    config = {
        "data_paths": {
            "processed_dir": str(tmp_path / "processed"),
            "artifacts_dir": str(tmp_path / "artifacts"),
        }
    }

    parquet_path, columns_path = save_feature_table(feature_df, config=config)

    # Verify parquet contains goal_diff and outcome
    loaded_parquet = pd.read_parquet(parquet_path)
    assert "goal_diff" in loaded_parquet.columns
    assert "outcome" in loaded_parquet.columns

    # Verify feature_columns.json does NOT contain goal_diff or outcome
    feature_cols = load_artifact("feature_columns.json", tmp_path / "artifacts")
    assert "goal_diff" not in feature_cols
    assert "outcome" not in feature_cols


def test_feature_columns_json_is_nonempty_list(
    sample_results_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test that feature_columns.json was created and is a non-empty list."""
    feature_df = build_feature_table(sample_results_df)

    config = {
        "data_paths": {
            "processed_dir": str(tmp_path / "processed"),
            "artifacts_dir": str(tmp_path / "artifacts"),
        }
    }

    parquet_path, columns_path = save_feature_table(feature_df, config=config)

    # Verify file exists
    assert columns_path.exists()

    # Load and verify it's a non-empty list
    feature_cols = load_artifact("feature_columns.json", tmp_path / "artifacts")
    assert isinstance(feature_cols, list)
    assert len(feature_cols) > 0

    # Verify it contains expected feature columns
    assert "home_elo_pre" in feature_cols
    assert "away_elo_pre" in feature_cols
    assert "elo_diff" in feature_cols
    assert "tournament_tier" in feature_cols
