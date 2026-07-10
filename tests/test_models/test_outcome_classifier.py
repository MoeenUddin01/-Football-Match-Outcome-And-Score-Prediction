"""Tests for the outcome classifier module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from football_predictor.models.outcome_classifier import (
    encode_tournament_tier,
    fit_tournament_encoder,
)
from football_predictor.utils.io import load_artifact, save_artifact


def test_encoder_handles_unseen_category() -> None:
    """Test that encoder handles unseen categories gracefully."""
    # Fit on training data with known categories
    train_tiers = pd.Series(["world_cup", "friendly", "qualifier", "world_cup", "friendly"])
    encoder = fit_tournament_encoder(train_tiers, config={"data_paths": {"artifacts_dir": "/tmp/test_artifacts"}})

    # Transform with unseen category
    test_tiers = pd.Series(["world_cup", "unseen_category", "friendly"])
    encoded = encode_tournament_tier(encoder, test_tiers)

    # Should not crash
    assert encoded.shape[0] == 3
    assert encoded.shape[1] == len(encoder.get_feature_names_out())

    # Unseen category should be all zeros
    unseen_row = encoded.iloc[1]
    assert unseen_row.sum() == 0.0


def test_encoder_produces_expected_columns() -> None:
    """Test that encoder produces expected column names."""
    train_tiers = pd.Series(["world_cup", "friendly", "qualifier"])
    encoder = fit_tournament_encoder(train_tiers, config={"data_paths": {"artifacts_dir": "/tmp/test_artifacts"}})

    encoded = encode_tournament_tier(encoder, train_tiers)
    expected_cols = encoder.get_feature_names_out(["tournament_tier"])
    assert list(encoded.columns) == list(expected_cols)


def test_model_save_reload_roundtrip() -> None:
    """Test that model produces identical predictions before and after save/reload."""
    from xgboost import XGBClassifier

    # Create a tiny trained model with integer labels (XGBoost requirement)
    X_train = np.random.rand(20, 5)
    y_train = np.random.choice([0, 1, 2], size=20)
    clf = XGBClassifier(
        objective="multi:softprob",
        n_estimators=10,
        max_depth=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train, verbose=False)

    # Get predictions before save
    X_test = np.random.rand(5, 5)
    probs_before = clf.predict_proba(X_test)

    # Save and reload
    save_artifact(clf, "test_model.pkl", "/tmp/test_models")
    clf_reloaded = load_artifact("test_model.pkl", "/tmp/test_models")

    # Get predictions after reload
    probs_after = clf_reloaded.predict_proba(X_test)

    # Predictions must be identical
    np.testing.assert_array_almost_equal(probs_before, probs_after)


def test_final_feature_columns_json() -> None:
    """Test that final_feature_columns.json is valid and excludes forbidden columns."""
    # Create a synthetic feature columns file
    feature_cols = [
        "home_elo_pre", "away_elo_pre", "elo_diff",
        "home_goals_scored_avg_last_5", "away_goals_scored_avg_last_5",
        "tournament_tier_world_cup", "tournament_tier_friendly",
    ]
    save_artifact(feature_cols, "test_feature_columns.json", "/tmp/test_artifacts")

    loaded = load_artifact("test_feature_columns.json", "/tmp/test_artifacts")

    assert isinstance(loaded, list)
    assert len(loaded) > 0
    forbidden = {"outcome", "goal_diff", "home_score", "away_score"}
    assert not forbidden.intersection(set(loaded))
