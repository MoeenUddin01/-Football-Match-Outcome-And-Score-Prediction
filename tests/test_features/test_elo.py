import logging

import pandas as pd
import pytest

from football_predictor.features.elo import (
    calculate_elo_update,
    compute_elo_ratings,
    get_goal_difference_multiplier,
    get_k_factor,
    get_rating_history,
    resolve_k_factor,
)


def test_calculate_elo_update_matches_spec_formula():
    home_new, away_new, home_expected, away_expected = calculate_elo_update(
        home_rating=1500.0,
        away_rating=1500.0,
        home_score=1,
        away_score=0,
        tournament="friendly",
        neutral=False,
        home_advantage=100.0,
        initial_rating=1500.0,
        k_factor=20.0,
    )

    assert home_expected == pytest.approx(0.6401, abs=1e-4)
    assert away_expected == pytest.approx(0.3599, abs=1e-4)
    assert home_new == pytest.approx(1507.2, abs=1e-1)
    assert away_new == pytest.approx(1492.8, abs=1e-1)


def test_compute_elo_ratings_uses_initial_rating_on_first_match():
    matches = pd.DataFrame(
        [
            {
                "date": "2023-01-01",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 1,
                "away_score": 0,
                "neutral": False,
                "tournament": "friendly",
            }
        ]
    )

    results = compute_elo_ratings(matches)

    assert results.iloc[0]["home_elo_pre"] == pytest.approx(1500.0)
    assert results.iloc[0]["away_elo_pre"] == pytest.approx(1500.0)


def test_compute_elo_ratings_processes_matches_chronologically():
    matches = pd.DataFrame(
        [
            {
                "date": "2023-01-01",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 1,
                "away_score": 0,
                "neutral": False,
                "tournament": "friendly",
            },
            {
                "date": "2023-02-01",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 0,
                "away_score": 1,
                "neutral": False,
                "tournament": "friendly",
            },
        ]
    )

    results = compute_elo_ratings(matches)

    assert results.iloc[0]["home_elo_pre"] == pytest.approx(1500.0)
    assert results.iloc[0]["away_elo_pre"] == pytest.approx(1500.0)
    assert results.iloc[0]["home_elo_post"] == pytest.approx(1503.6, abs=1e-1)
    assert results.iloc[0]["away_elo_post"] == pytest.approx(1496.4, abs=1e-1)

    assert results.iloc[1]["home_elo_pre"] == pytest.approx(1503.6, abs=1e-1)
    assert results.iloc[1]["away_elo_pre"] == pytest.approx(1496.4, abs=1e-1)


def test_compute_elo_ratings_requires_sorted_input():
    matches = pd.DataFrame(
        [
            {
                "date": "2023-02-01",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 1,
                "away_score": 0,
                "neutral": False,
                "tournament": "friendly",
            },
            {
                "date": "2023-01-01",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 0,
                "away_score": 1,
                "neutral": False,
                "tournament": "friendly",
            },
        ]
    )

    with pytest.raises(ValueError):
        compute_elo_ratings(matches)


def test_goal_difference_multiplier_matches_spec():
    assert get_goal_difference_multiplier(1) == pytest.approx(1.0)
    assert get_goal_difference_multiplier(2) == pytest.approx(1.5)
    assert get_goal_difference_multiplier(4) == pytest.approx(1.875)


def test_get_k_factor_uses_tier_lookup_and_warns_for_default(caplog):
    caplog.set_level(logging.WARNING)

    assert get_k_factor("Friendly") == 10
    assert get_k_factor("FIFA World Cup") == 60
    assert get_k_factor("UEFA Euro") == 50
    assert get_k_factor("Some Random Cup") == 20

    assert "Some Random Cup" in caplog.text


def test_get_rating_history_returns_team_ratings_over_time():
    matches = pd.DataFrame(
        [
            {
                "date": "2023-01-01",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 1,
                "away_score": 0,
                "neutral": False,
                "tournament": "friendly",
                "home_elo_post": 1507.2,
                "away_elo_post": 1492.8,
            },
            {
                "date": "2023-02-01",
                "home_team": "Team A",
                "away_team": "Team C",
                "home_score": 0,
                "away_score": 1,
                "neutral": False,
                "tournament": "friendly",
                "home_elo_post": 1498.0,
                "away_elo_post": 1502.0,
            },
        ]
    )

    history = get_rating_history("Team A", matches)

    assert list(history["rating"]) == pytest.approx([1507.2, 1498.0])
    assert list(history["date"]) == [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")]


def test_resolve_k_factor_is_alias_for_get_k_factor():
    assert resolve_k_factor("friendly") == 10
