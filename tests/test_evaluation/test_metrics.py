"""Tests for the evaluation metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_predictor.evaluation.metrics import (
    accuracy,
    brier_score,
    log_loss_score,
    predict_to_labels,
)


def test_log_loss_known_answer() -> None:
    """Test log loss with a known-answer case."""
    # Perfect prediction: log_loss should be very low
    y_true = pd.Series(["home_win", "draw", "away_win"])
    y_pred_probs = pd.DataFrame({
        "home_win_prob": [1.0, 0.0, 0.0],
        "draw_prob": [0.0, 1.0, 0.0],
        "away_win_prob": [0.0, 0.0, 1.0],
    })

    result = log_loss_score(y_true, y_pred_probs)
    assert result == pytest.approx(0.0, abs=0.001)


def test_log_loss_uniform_prediction() -> None:
    """Test log loss with uniform predictions."""
    y_true = pd.Series(["home_win", "draw", "away_win"])
    y_pred_probs = pd.DataFrame({
        "home_win_prob": [1/3, 1/3, 1/3],
        "draw_prob": [1/3, 1/3, 1/3],
        "away_win_prob": [1/3, 1/3, 1/3],
    })

    result = log_loss_score(y_true, y_pred_probs)
    expected = -np.log(1/3)
    assert result == pytest.approx(expected, abs=0.001)


def test_brier_score_perfect_prediction() -> None:
    """Test Brier score with perfect predictions."""
    y_true = pd.Series(["home_win", "draw", "away_win"])
    y_pred_probs = pd.DataFrame({
        "home_win_prob": [1.0, 0.0, 0.0],
        "draw_prob": [0.0, 1.0, 0.0],
        "away_win_prob": [0.0, 0.0, 1.0],
    })

    result = brier_score(y_true, y_pred_probs)
    assert result == pytest.approx(0.0, abs=0.001)


def test_brier_score_uniform_prediction() -> None:
    """Test Brier score with uniform predictions."""
    y_true = pd.Series(["home_win", "draw", "away_win"])
    y_pred_probs = pd.DataFrame({
        "home_win_prob": [1/3, 1/3, 1/3],
        "draw_prob": [1/3, 1/3, 1/3],
        "away_win_prob": [1/3, 1/3, 1/3],
    })

    result = brier_score(y_true, y_pred_probs)
    # For uniform predictions on 3 classes: (1-1/3)^2 + (0-1/3)^2 + (0-1/3)^2 per sample
    # = 4/9 + 1/9 + 1/9 = 6/9 = 2/3 per sample
    expected = 2/3
    assert result == pytest.approx(expected, abs=0.001)


def test_accuracy_known_answer() -> None:
    """Test accuracy with known predictions."""
    y_true = pd.Series(["home_win", "draw", "away_win", "home_win"])
    y_pred_labels = pd.Series(["home_win", "draw", "draw", "home_win"])

    result = accuracy(y_true, y_pred_labels)
    assert result == pytest.approx(0.75, abs=0.001)


def test_predict_to_labels() -> None:
    """Test conversion of probabilities to labels."""
    probs_df = pd.DataFrame({
        "home_win_prob": [0.7, 0.2, 0.1],
        "draw_prob": [0.2, 0.6, 0.3],
        "away_win_prob": [0.1, 0.2, 0.6],
    })

    labels = predict_to_labels(probs_df)
    expected = pd.Series(["home_win", "draw", "away_win"])
    pd.testing.assert_series_equal(labels, expected)
