"""Tests for the Elo baseline model."""

from __future__ import annotations

import pandas as pd
import pytest

from football_predictor.models.baseline_elo import (
    compute_draw_rate,
    compute_elo_expected_score,
    predict_elo_baseline,
)


def test_compute_elo_expected_score_known_answer() -> None:
    """Test elo expected score with a hand-computed example."""
    # Two teams at equal rating, non-neutral: home advantage = 100
    # rating_difference = 100
    # expected_home = 1 / (10^(-100/400) + 1) = 1 / (10^(-0.25) + 1)
    # 10^(-0.25) ≈ 0.5623
    # expected_home ≈ 1 / (0.5623 + 1) ≈ 0.6401
    we_home = compute_elo_expected_score(0, neutral=False, home_advantage=100)
    assert we_home == pytest.approx(0.6401, abs=0.001)


def test_compute_elo_expected_score_neutral() -> None:
    """Test elo expected score at neutral venue."""
    # Two teams at equal rating, neutral: no home advantage
    # expected_home = 1 / (10^(0) + 1) = 0.5
    we_home = compute_elo_expected_score(0, neutral=True, home_advantage=100)
    assert we_home == pytest.approx(0.5, abs=0.001)


def test_predict_elo_baseline_known_answer() -> None:
    """Test predict_elo_baseline with hand-computed probabilities."""
    # elo_diff = 0, non-neutral, draw_rate = 0.25
    # we_home = 0.6401, we_away = 0.3599
    # p_home_win = (1 - 0.25) * (0.6401 / 1.0) = 0.4801
    # p_draw = 0.25
    # p_away_win = (1 - 0.25) * (0.3599 / 1.0) = 0.2699
    result = predict_elo_baseline(0, neutral=False, draw_rate=0.25)

    assert result["home_win_prob"] == pytest.approx(0.4801, abs=0.001)
    assert result["draw_prob"] == pytest.approx(0.25, abs=0.001)
    assert result["away_win_prob"] == pytest.approx(0.2699, abs=0.001)


def test_compute_draw_rate() -> None:
    """Test draw rate computation."""
    outcomes = ["home_win", "draw", "away_win", "draw", "home_win"]
    draw_rate = compute_draw_rate(pd.Series(outcomes))
    assert draw_rate == pytest.approx(0.4, abs=0.001)


def test_probabilities_sum_to_one() -> None:
    """Test that probabilities sum to 1.0."""
    result = predict_elo_baseline(100, neutral=False, draw_rate=0.3)
    total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
    assert total == pytest.approx(1.0, abs=0.001)
