"""Score regressor using Poisson regression for home/away goal prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from football_predictor.models.outcome_classifier import (
    _build_feature_matrix,
    _load_config,
)
from football_predictor.utils.io import load_artifact, save_artifact

logger = logging.getLogger(__name__)

MAX_GOALS = 10


def train_score_regressors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    config: Any | None = None,
) -> tuple[PoissonRegressor, PoissonRegressor, StandardScaler]:
    """Train two Poisson regressors for home and away goal prediction.

    Reuses the same feature matrix construction as the outcome classifier:
    same encoder, same final_feature_columns.json, ensuring identical inputs.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with feature columns, 'home_score', and 'away_score'.
    val_df : pd.DataFrame
        Validation data.
    feature_columns : list[str]
        Base numeric feature column names (excluding tournament_tier).
        Not used directly — loaded from saved artifacts.
    config : dict, optional
        Configuration dictionary.

    Returns
    -------
    tuple[PoissonRegressor, PoissonRegressor, StandardScaler]
        Trained (home_goals_model, away_goals_model, feature_scaler).
    """
    config_data = _load_config(config)
    model_params = config_data.get("model_params", {})
    artifacts_dir = config_data.get("data_paths", {}).get("artifacts_dir", "outputs/artifacts")
    models_dir = config_data.get("data_paths", {}).get("models_dir", "outputs/models")

    # Reuse the saved encoder and final_feature_columns from the classifier
    encoder = load_artifact("tournament_tier_encoder.pkl", artifacts_dir)
    final_feature_columns = load_artifact("final_feature_columns.json", artifacts_dir)

    numeric_cols = [col for col in final_feature_columns if not col.startswith("tournament_tier_")]

    X_train = _build_feature_matrix(train_df, encoder, numeric_cols, final_feature_columns)
    X_val = _build_feature_matrix(val_df, encoder, numeric_cols, final_feature_columns)

    # Standardize features — Poisson regression (L-BFGS) needs comparable scales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    y_home_train = train_df["home_score"].values.astype(float)
    y_away_train = train_df["away_score"].values.astype(float)
    y_home_val = val_df["home_score"].values.astype(float)
    y_away_val = val_df["away_score"].values.astype(float)

    alpha = model_params.get("poisson_alpha", 1.0)

    logger.info("Training Poisson regressors with alpha=%.2f", alpha)

    # Home goals model
    home_model = PoissonRegressor(alpha=alpha, max_iter=1000)
    home_model.fit(X_train_scaled, y_home_train)
    home_val_pred = home_model.predict(X_val_scaled)
    home_mae = float(np.mean(np.abs(home_val_pred - y_home_val)))
    logger.info("Home goals model val MAE: %.4f", home_mae)

    # Away goals model
    away_model = PoissonRegressor(alpha=alpha, max_iter=1000)
    away_model.fit(X_train_scaled, y_away_train)
    away_val_pred = away_model.predict(X_val_scaled)
    away_mae = float(np.mean(np.abs(away_val_pred - y_away_val)))
    logger.info("Away goals model val MAE: %.4f", away_mae)

    # Save models and scaler
    save_artifact(home_model, "home_score_regressor.pkl", models_dir)
    save_artifact(away_model, "away_score_regressor.pkl", models_dir)
    save_artifact(scaler, "score_regressor_scaler.pkl", artifacts_dir)
    logger.info("Saved score regressors to %s", models_dir)

    return home_model, away_model, scaler


def poisson_to_outcome_probs(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = MAX_GOALS,
) -> dict[str, float]:
    """Derive win/draw/loss probabilities from Poisson goal rate predictions.

    Uses the Poisson distribution to sum over all reasonable scoreline
    combinations (0-0 through max_goals-max_goals) and compute:
    P(home_win), P(draw), P(away_win).

    Parameters
    ----------
    home_lambda : float
        Predicted expected home goals (must be >= 0).
    away_lambda : float
        Predicted expected away goals (must be >= 0).
    max_goals : int, optional
        Maximum number of goals to consider per team (default 10).

    Returns
    -------
    dict
        Keys: 'home_win_prob', 'draw_prob', 'away_win_prob'.
    """
    home_lambda = max(float(home_lambda), 1e-10)
    away_lambda = max(float(away_lambda), 1e-10)

    home_goals = np.arange(0, max_goals + 1)
    away_goals = np.arange(0, max_goals + 1)

    home_pmf = poisson.pmf(home_goals, home_lambda)
    away_pmf = poisson.pmf(away_goals, away_lambda)

    # Outer product gives P(home=i, away=j) for all i,j combinations
    score_matrix = np.outer(home_pmf, away_pmf)

    # Home win: home_goals > away_goals (lower triangle, i > j)
    home_win_prob = float(np.sum(np.tril(score_matrix, k=-1)))
    draw_prob = float(np.sum(np.diag(score_matrix)))
    # Away win: away_goals > home_goals (upper triangle, j > i)
    away_win_prob = float(np.sum(np.triu(score_matrix, k=1)))

    # Normalize to handle truncation at max_goals
    total = home_win_prob + draw_prob + away_win_prob
    home_win_prob /= total
    draw_prob /= total
    away_win_prob /= total

    return {
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
    }


def evaluate_score_regressors(
    home_model: PoissonRegressor,
    away_model: PoissonRegressor,
    scaler: StandardScaler,
    val_df: pd.DataFrame,
    final_feature_columns: list[str],
    train_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Evaluate score regressors and compare with naive average baseline.

    Parameters
    ----------
    home_model : PoissonRegressor
        Trained home goals model.
    away_model : PoissonRegressor
        Trained away goals model.
    scaler : StandardScaler
        Fitted feature scaler.
    val_df : pd.DataFrame
        Validation data.
    final_feature_columns : list[str]
        Exact ordered feature columns.
    train_df : pd.DataFrame
        Training data (for computing naive baseline averages).

    Returns
    -------
    dict
        Dictionary with 'poisson' and 'naive_baseline' MAE results.
    """
    config_data = _load_config()
    artifacts_dir = config_data.get("data_paths", {}).get("artifacts_dir", "outputs/artifacts")
    encoder = load_artifact("tournament_tier_encoder.pkl", artifacts_dir)
    numeric_cols = [col for col in final_feature_columns if not col.startswith("tournament_tier_")]
    X_val = _build_feature_matrix(val_df, encoder, numeric_cols, final_feature_columns)
    X_val_scaled = scaler.transform(X_val)

    # Poisson predictions
    home_pred = home_model.predict(X_val_scaled)
    away_pred = away_model.predict(X_val_scaled)

    home_mae = float(np.mean(np.abs(home_pred - val_df["home_score"].values)))
    away_mae = float(np.mean(np.abs(away_pred - val_df["away_score"].values)))

    # Naive baseline: always predict training set average
    naive_home = float(train_df["home_score"].mean())
    naive_away = float(train_df["away_score"].mean())
    naive_home_mae = float(np.mean(np.abs(naive_home - val_df["home_score"].values)))
    naive_away_mae = float(np.mean(np.abs(naive_away - val_df["away_score"].values)))

    results = {
        "poisson": {"home_mae": home_mae, "away_mae": away_mae},
        "naive_baseline": {"home_mae": naive_home_mae, "away_mae": naive_away_mae},
    }

    print("\n" + "=" * 70)
    print("SCORE REGRESSOR COMPARISON — VALIDATION SET (MAE)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Poisson':>15} {'Naive Average':>15}")
    print("-" * 70)
    print(f"{'Home goals MAE':<25} {home_mae:>15.4f} {naive_home_mae:>15.4f}")
    print(f"{'Away goals MAE':<25} {away_mae:>15.4f} {naive_away_mae:>15.4f}")
    print("=" * 70)

    return results
