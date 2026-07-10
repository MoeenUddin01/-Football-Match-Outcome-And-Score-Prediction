"""Tests for the score regressor module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import PoissonRegressor

from football_predictor.models.score_regressor import poisson_to_outcome_probs
from football_predictor.utils.io import load_artifact, save_artifact


def test_poisson_to_outcome_probs_sums_to_one() -> None:
    """Test that outcome probabilities sum to approximately 1.0."""
    probs = poisson_to_outcome_probs(home_lambda=1.3, away_lambda=1.1)
    total = probs["home_win_prob"] + probs["draw_prob"] + probs["away_win_prob"]
    assert total == pytest.approx(1.0, abs=0.001)


def test_poisson_to_outcome_probs_even_match_draw_higher() -> None:
    """Test that evenly matched teams have higher draw probability than mismatched."""
    # Evenly matched: similar lambda
    even = poisson_to_outcome_probs(home_lambda=1.5, away_lambda=1.5)

    # Mismatched: very different lambdas
    mismatched = poisson_to_outcome_probs(home_lambda=2.5, away_lambda=0.5)

    assert even["draw_prob"] > mismatched["draw_prob"]


def test_poisson_to_outcome_probs_strong_favorite() -> None:
    """Test that a strong home favorite has high home_win probability."""
    probs = poisson_to_outcome_probs(home_lambda=3.0, away_lambda=0.5)
    assert probs["home_win_prob"] > 0.7


def test_poisson_to_outcome_probs_zero_goals() -> None:
    """Test that very low lambdas concentrate probability in draw."""
    probs = poisson_to_outcome_probs(home_lambda=0.01, away_lambda=0.01)
    assert probs["draw_prob"] > 0.9


def test_models_train_and_predict_nonnegative() -> None:
    """Test that both Poisson models train and produce non-negative predictions."""
    # Create synthetic training data
    np.random.seed(42)
    n = 100
    X = np.random.rand(n, 5)
    y_home = np.random.poisson(lam=1.5, size=n).astype(float)
    y_away = np.random.poisson(lam=1.1, size=n).astype(float)

    home_model = PoissonRegressor(alpha=1.0, max_iter=1000)
    home_model.fit(X, y_home)
    home_pred = home_model.predict(X)

    away_model = PoissonRegressor(alpha=1.0, max_iter=1000)
    away_model.fit(X, y_away)
    away_pred = away_model.predict(X)

    # Poisson naturally guarantees non-negative predictions
    assert np.all(home_pred >= 0)
    assert np.all(away_pred >= 0)


def test_model_save_reload_roundtrip() -> None:
    """Test that model produces identical predictions before and after save/reload."""
    np.random.seed(42)
    n = 50
    X = np.random.rand(n, 3)
    y = np.random.poisson(lam=1.5, size=n).astype(float)

    model = PoissonRegressor(alpha=1.0, max_iter=1000)
    model.fit(X, y)

    X_test = np.random.rand(5, 3)
    pred_before = model.predict(X_test)

    save_artifact(model, "test_poisson.pkl", "/tmp/test_models")
    model_reloaded = load_artifact("test_poisson.pkl", "/tmp/test_models")
    pred_after = model_reloaded.predict(X_test)

    np.testing.assert_array_almost_equal(pred_before, pred_after)
