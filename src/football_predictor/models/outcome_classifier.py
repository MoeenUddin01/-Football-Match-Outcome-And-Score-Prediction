"""Outcome classifier using XGBoost with tournament tier one-hot encoding."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

from football_predictor.evaluation.metrics import (
    accuracy,
    brier_score,
    log_loss_score,
    predict_to_labels,
)
from football_predictor.utils.io import load_artifact, save_artifact

logger = logging.getLogger(__name__)

LABEL_ORDER = ["away_win", "draw", "home_win"]


def _load_config(config: Any | None = None) -> dict[str, Any]:
    """Load config from a dict or the repository YAML file."""
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


def fit_tournament_encoder(
    train_tiers: pd.Series,
    config: Any | None = None,
) -> OneHotEncoder:
    """Fit a one-hot encoder on training tournament tiers and save it.

    Parameters
    ----------
    train_tiers : pd.Series
        Tournament tier labels from training data only.
    config : dict, optional
        Configuration dictionary.

    Returns
    -------
    OneHotEncoder
        Fitted encoder.
    """
    config_data = _load_config(config)
    artifacts_dir = config_data.get("data_paths", {}).get("artifacts_dir", "outputs/artifacts")

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        dtype=float,
    )
    encoder.fit(np.asarray(train_tiers).reshape(-1, 1))

    save_artifact(encoder, "tournament_tier_encoder.pkl", artifacts_dir)
    logger.info("Saved tournament tier encoder to %s", artifacts_dir)

    return encoder


def encode_tournament_tier(
    encoder: OneHotEncoder,
    tiers: pd.Series,
) -> pd.DataFrame:
    """Transform tournament tiers using a fitted encoder.

    Parameters
    ----------
    encoder : OneHotEncoder
        Fitted encoder.
    tiers : pd.Series
        Tournament tier labels to encode.

    Returns
    -------
    pd.DataFrame
        One-hot encoded columns with names like tournament_tier_<name>.
    """
    encoded = encoder.transform(np.asarray(tiers).reshape(-1, 1))
    feature_names = encoder.get_feature_names_out(["tournament_tier"])
    return pd.DataFrame(encoded, columns=feature_names, index=tiers.index)


def _build_feature_matrix(
    df: pd.DataFrame,
    encoder: OneHotEncoder,
    numeric_cols: list[str],
    final_feature_columns: list[str],
) -> pd.DataFrame:
    """Build the final feature matrix from a dataframe."""
    encoded = encode_tournament_tier(encoder, df["tournament_tier"])
    X = pd.concat([df[numeric_cols].reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    return X[final_feature_columns]


def train_outcome_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    config: Any | None = None,
) -> XGBClassifier:
    """Train an XGBoost multi-class classifier for match outcome prediction.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with feature columns and 'outcome' column.
    val_df : pd.DataFrame
        Validation data with feature columns and 'outcome' column.
    feature_columns : list[str]
        Base numeric feature column names (excluding tournament_tier).
    config : dict, optional
        Configuration dictionary.

    Returns
    -------
    XGBClassifier
        Trained classifier.
    """
    config_data = _load_config(config)
    model_params = config_data.get("model_params", {})
    random_seed = config_data.get("random_seed", 42)
    artifacts_dir = config_data.get("data_paths", {}).get("artifacts_dir", "outputs/artifacts")
    models_dir = config_data.get("data_paths", {}).get("models_dir", "outputs/models")

    # Part A: Fit and apply tournament tier encoder
    encoder = fit_tournament_encoder(train_df["tournament_tier"], config=config_data)

    # Build final feature columns (numeric + one-hot encoded tournament tier)
    numeric_cols = [col for col in feature_columns if col != "tournament_tier"]
    encoded_cols = list(encoder.get_feature_names_out(["tournament_tier"]))
    final_feature_columns = numeric_cols + encoded_cols

    X_train = _build_feature_matrix(train_df, encoder, numeric_cols, final_feature_columns)
    X_val = _build_feature_matrix(val_df, encoder, numeric_cols, final_feature_columns)

    # Encode string labels to integers for XGBoost 3.x
    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y_train_int = le.transform(train_df["outcome"].values)
    y_val_int = le.transform(val_df["outcome"].values)

    logger.info("Training XGBoost classifier with params: %s", model_params)
    logger.info("Feature columns (%d): %s", len(final_feature_columns), final_feature_columns)

    # Part B: Train classifier
    clf = XGBClassifier(
        objective="multi:softprob",
        max_depth=model_params.get("max_depth", 5),
        learning_rate=model_params.get("learning_rate", 0.05),
        n_estimators=model_params.get("n_estimators", 300),
        random_state=random_seed,
        eval_metric="mlogloss",
    )

    clf.fit(X_train, y_train_int, eval_set=[(X_val, y_val_int)], verbose=False)

    # Save model and feature columns
    save_artifact(clf, "outcome_classifier.pkl", models_dir)
    save_artifact(final_feature_columns, "final_feature_columns.json", artifacts_dir)
    logger.info("Saved model to %s", models_dir)
    logger.info("Saved final feature columns to %s", artifacts_dir)

    return clf


def evaluate_classifier(
    clf: XGBClassifier,
    val_df: pd.DataFrame,
    final_feature_columns: list[str],
    elo_baseline_probs: pd.DataFrame,
    elo_baseline_labels: pd.Series,
    y_true: pd.Series,
    config: Any | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate classifier and compare with Elo baseline.

    Parameters
    ----------
    clf : XGBClassifier
        Trained classifier.
    val_df : pd.DataFrame
        Validation data.
    final_feature_columns : list[str]
        Exact ordered feature columns the model expects.
    elo_baseline_probs : pd.DataFrame
        Elo baseline predictions.
    elo_baseline_labels : pd.Series
        Elo baseline predicted labels.
    y_true : pd.Series
        True outcome labels.
    config : dict, optional
        Configuration dictionary.

    Returns
    -------
    dict
        Dictionary with 'elo_baseline' and 'xgboost' metric results.
    """
    config_data = _load_config(config)
    artifacts_dir = config_data.get("data_paths", {}).get("artifacts_dir", "outputs/artifacts")
    encoder = load_artifact("tournament_tier_encoder.pkl", artifacts_dir)

    # Derive numeric columns by excluding one-hot tournament_tier columns
    numeric_cols = [col for col in final_feature_columns if not col.startswith("tournament_tier_")]
    X_val = _build_feature_matrix(val_df, encoder, numeric_cols, final_feature_columns)

    # Column ordering must match training exactly
    xgb_probs_raw = clf.predict_proba(X_val)
    xgb_probs = pd.DataFrame(
        xgb_probs_raw,
        columns=["away_win_prob", "draw_prob", "home_win_prob"],
        index=val_df.index,
    )
    xgb_labels = predict_to_labels(xgb_probs)

    results = {
        "elo_baseline": {
            "log_loss": log_loss_score(y_true, elo_baseline_probs),
            "brier": brier_score(y_true, elo_baseline_probs),
            "accuracy": accuracy(y_true, elo_baseline_labels),
        },
        "xgboost": {
            "log_loss": log_loss_score(y_true, xgb_probs),
            "brier": brier_score(y_true, xgb_probs),
            "accuracy": accuracy(y_true, xgb_labels),
        },
    }

    print("\n" + "=" * 70)
    print("MODEL COMPARISON — VALIDATION SET")
    print("=" * 70)
    print(f"{'Metric':<15} {'Elo Baseline':>15} {'XGBoost':>15}")
    print("-" * 70)
    for metric_name in ["log_loss", "brier", "accuracy"]:
        elo_val = results["elo_baseline"][metric_name]
        xgb_val = results["xgboost"][metric_name]
        print(f"{metric_name:<15} {elo_val:>15.4f} {xgb_val:>15.4f}")
    print("=" * 70)

    return results
