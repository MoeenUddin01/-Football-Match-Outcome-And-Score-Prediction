"""Tests for the rolling stats module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.football_predictor.features.rolling_stats import compute_rolling_features


@pytest.fixture
def simple_results_df() -> pd.DataFrame:
    """Create a minimal results DataFrame for testing."""
    data = {
        "date": [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-05",
            "2020-01-06",
        ],
        "home_team": ["Brazil", "Argentina", "Germany", "France", "Spain", "Brazil"],
        "away_team": ["Argentina", "Germany", "France", "Spain", "Brazil", "Argentina"],
        "home_score": [2, 1, 0, 3, 1, 1],
        "away_score": [1, 1, 2, 1, 2, 0],
    }
    return pd.DataFrame(data)


def test_known_answer_goals_scored_avg_last_3() -> None:
    """Test that goals_scored_avg_last_3 matches hand-computed values."""
    data = {
        "date": [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
        ],
        "home_team": ["Brazil", "Argentina", "Brazil", "Argentina"],
        "away_team": ["Argentina", "Brazil", "Argentina", "Brazil"],
        "home_score": [2, 1, 3, 0],
        "away_score": [1, 2, 0, 1],
    }
    df = pd.DataFrame(data)
    result = compute_rolling_features(df, window_sizes={"goals_window": 3, "form_window": 3, "h2h_window": 3})

    # Match 0 (2020-01-01): Brazil at home, first match -> 0
    assert result.iloc[0]["home_goals_scored_avg_last_N"] == 0.0

    # Match 1 (2020-01-02): Argentina home vs Brazil
    # Brazil played once before (match 0) as home, scored 2 goals
    assert result.iloc[1]["away_goals_scored_avg_last_N"] == 2.0

    # Match 2 (2020-01-03): Brazil home vs Argentina
    # Brazil played twice before: match 0 (home, scored 2), match 1 (away, scored 2)
    # goals_scored_avg_last_3 = (2 + 2) / 2 = 2.0
    assert result.iloc[2]["home_goals_scored_avg_last_N"] == 2.0

    # Match 3 (2020-01-04): Argentina home vs Brazil
    # Brazil played 3 times before: match 0 (home, scored 2), match 1 (away, scored 2), match 2 (home, scored 3)
    # goals_scored_avg_last_3 = (2 + 2 + 3) / 3 = 2.333...
    assert result.iloc[3]["away_goals_scored_avg_last_N"] == pytest.approx(2.333, abs=0.01)


def test_leakage_test_match_n_does_not_affect_match_n() -> None:
    """Test that modifying match N's score does not affect rolling features at match N."""
    data_original = {
        "date": [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
        ],
        "home_team": ["Brazil", "Argentina", "Brazil", "Argentina"],
        "away_team": ["Argentina", "Brazil", "Argentina", "Brazil"],
        "home_score": [2, 1, 3, 0],
        "away_score": [1, 2, 0, 1],
    }
    df_original = pd.DataFrame(data_original)
    result_original = compute_rolling_features(df_original, window_sizes={"goals_window": 3, "form_window": 3, "h2h_window": 3})

    data_modified = {
        "date": [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
        ],
        "home_team": ["Brazil", "Argentina", "Brazil", "Argentina"],
        "away_team": ["Argentina", "Brazil", "Argentina", "Brazil"],
        "home_score": [2, 1, 99, 0],
        "away_score": [1, 2, 0, 1],
    }
    df_modified = pd.DataFrame(data_modified)
    result_modified = compute_rolling_features(df_modified, window_sizes={"goals_window": 3, "form_window": 3, "h2h_window": 3})

    # Match 2's score changed from 3-0 to 99-0
    # But match 2's own rolling features should NOT change
    assert result_original.iloc[2]["home_goals_scored_avg_last_N"] == result_modified.iloc[2]["home_goals_scored_avg_last_N"]
    assert result_original.iloc[2]["home_goals_conceded_avg_last_N"] == result_modified.iloc[2]["home_goals_conceded_avg_last_N"]
    assert result_original.iloc[2]["home_win_rate_last_N"] == result_modified.iloc[2]["home_win_rate_last_N"]

    # Match 3's features SHOULD change because they depend on match 2
    assert result_original.iloc[3]["away_goals_scored_avg_last_N"] != result_modified.iloc[3]["away_goals_scored_avg_last_N"]


def test_first_match_gets_defaults() -> None:
    """Test that a team's first appearance gets default values (0 for rolling, 30 for rest_days)."""
    data = {
        "date": [
            "2020-01-01",
            "2020-01-02",
        ],
        "home_team": ["Brazil", "Argentina"],
        "away_team": ["Argentina", "Brazil"],
        "home_score": [2, 1],
        "away_score": [1, 0],
    }
    df = pd.DataFrame(data)
    result = compute_rolling_features(df, window_sizes={"goals_window": 3, "form_window": 3, "h2h_window": 3})

    # Match 0: Both teams are playing for the first time
    assert result.iloc[0]["home_goals_scored_avg_last_N"] == 0.0
    assert result.iloc[0]["home_goals_conceded_avg_last_N"] == 0.0
    assert result.iloc[0]["home_win_rate_last_N"] == 0.0
    assert result.iloc[0]["home_rest_days"] == 30

    assert result.iloc[0]["away_goals_scored_avg_last_N"] == 0.0
    assert result.iloc[0]["away_goals_conceded_avg_last_N"] == 0.0
    assert result.iloc[0]["away_win_rate_last_N"] == 0.0
    assert result.iloc[0]["away_rest_days"] == 30


def test_head_to_head_no_prior_meetings() -> None:
    """Test that teams with no prior meetings get all zeros for h2h features."""
    data = {
        "date": [
            "2020-01-01",
        ],
        "home_team": ["Brazil"],
        "away_team": ["Argentina"],
        "home_score": [2],
        "away_score": [1],
    }
    df = pd.DataFrame(data)
    result = compute_rolling_features(df, window_sizes={"goals_window": 3, "form_window": 3, "h2h_window": 3})

    # First meeting between these two teams
    assert result.iloc[0]["h2h_home_wins"] == 0
    assert result.iloc[0]["h2h_away_wins"] == 0
    assert result.iloc[0]["h2h_draws"] == 0


def test_head_to_head_after_one_meeting() -> None:
    """Test that h2h features correctly reflect one prior meeting."""
    data = {
        "date": [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
        ],
        "home_team": ["Brazil", "Argentina", "Brazil"],
        "away_team": ["Argentina", "Brazil", "Argentina"],
        "home_score": [2, 1, 0],
        "away_score": [1, 2, 0],
    }
    df = pd.DataFrame(data)
    result = compute_rolling_features(df, window_sizes={"goals_window": 3, "form_window": 3, "h2h_window": 3})

    # Match 0: No prior meetings
    assert result.iloc[0]["h2h_home_wins"] == 0
    assert result.iloc[0]["h2h_away_wins"] == 0
    assert result.iloc[0]["h2h_draws"] == 0

    # Match 1: Argentina home vs Brazil (match 0 was Brazil home)
    # In match 0: Brazil won 2-1 (home win for Brazil)
    # Match 1 is Argentina home: home_score=1, away_score=2 -> Brazil won 2-1 (away win for Brazil)
    # So h2h_home_wins = Argentina's home wins = 0
    # h2h_away_wins = Brazil's away wins = 1
    assert result.iloc[1]["h2h_home_wins"] == 0
    assert result.iloc[1]["h2h_away_wins"] == 1
    assert result.iloc[1]["h2h_draws"] == 0

    # Match 2: Brazil home vs Argentina
    # Prior meetings: match 0 (Brazil won 2-1 at home), match 1 (Brazil won 2-1 away)
    # For match 2 (Brazil home), we count wins for the current home team (Brazil):
    # - Match 0: Brazil won at home -> counts as a home win
    # - Match 1: Brazil won away -> also counts as a home win for the current match
    #   (because Brazil won, and Brazil is the current home team)
    # So h2h_home_wins = 2 (Brazil won both previous meetings)
    # h2h_away_wins = 0 (Argentina won none)
    assert result.iloc[2]["h2h_home_wins"] == 2
    assert result.iloc[2]["h2h_away_wins"] == 0
    assert result.iloc[2]["h2h_draws"] == 0
