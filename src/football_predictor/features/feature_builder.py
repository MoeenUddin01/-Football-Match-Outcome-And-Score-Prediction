"""Assemble the final feature table from Elo and rolling statistics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from football_predictor.data.cleaner import clean_results
from football_predictor.data.loader import load_results
from football_predictor.features.elo import _load_config, _resolve_tier_name, compute_elo_ratings
from football_predictor.features.rolling_stats import compute_rolling_features
from football_predictor.utils.io import save_artifact

logger = logging.getLogger(__name__)


def _load_config_for_builder(config: Any | None = None) -> dict[str, Any]:
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


def build_feature_table(
    results_df: pd.DataFrame,
    config: Any | None = None,
) -> pd.DataFrame:
    """Build the complete feature table from raw match results.

    Steps:
    1. Compute Elo ratings (chronological forward-pass)
    2. Compute rolling features (chronological forward-pass)
    3. Merge outputs (row-aligned verification via assertion)
    4. Add derived columns: elo_diff, goal_diff, outcome
    5. Keep identifier/context columns
    6. Encode tournament into tournament_tier

    Parameters
    ----------
    results_df : pd.DataFrame
        Cleaned match results sorted by date.
    config : dict, optional
        Configuration dictionary. If None, loads from config.yaml.

    Returns
    -------
    pd.DataFrame
        Complete feature table with one row per match.
    """
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame")

    config_data = _load_config_for_builder(config)

    logger.info("Building feature table from %d matches", len(results_df))

    logger.info("Step 1: Computing Elo ratings...")
    elo_df = compute_elo_ratings(results_df, config=config_data)

    logger.info("Step 2: Computing rolling features...")
    rolling_df = compute_rolling_features(results_df, config=config_data)

    logger.info("Step 3: Merging outputs...")
    assert len(elo_df) == len(rolling_df), (
        f"Row count mismatch: Elo output has {len(elo_df)} rows, "
        f"rolling output has {len(rolling_df)} rows"
    )

    if not elo_df.index.equals(rolling_df.index):
        logger.warning("Index mismatch between Elo and rolling outputs; aligning by position")
        rolling_df = rolling_df.reset_index(drop=True)

    feature_df = elo_df.copy()

    rolling_cols_to_add = [col for col in rolling_df.columns if col not in elo_df.columns]
    for col in rolling_cols_to_add:
        feature_df[col] = rolling_df[col].values

    logger.info("Step 4: Adding derived columns...")
    feature_df["elo_diff"] = feature_df["home_elo_pre"] - feature_df["away_elo_pre"]

    # goal_diff is ONLY used to derive the outcome label, NOT as a model feature
    feature_df["goal_diff"] = feature_df["home_score"] - feature_df["away_score"]

    def _determine_outcome(goal_diff: float) -> str:
        if goal_diff > 0:
            return "home_win"
        elif goal_diff < 0:
            return "away_win"
        return "draw"

    feature_df["outcome"] = feature_df["goal_diff"].apply(_determine_outcome)

    logger.info("Step 5: Retaining identifier columns...")
    identifier_cols = [
        "date",
        "home_team",
        "away_team",
        "tournament",
        "neutral",
        "home_score",
        "away_score",
    ]

    logger.info("Step 6: Encoding tournament tiers...")
    feature_df["tournament_tier"] = feature_df["tournament"].apply(
        lambda t: _resolve_tier_name(str(t))
    )

    all_cols = list(feature_df.columns)
    logger.info("Feature table has %d columns: %s", len(all_cols), all_cols)

    return feature_df


def save_feature_table(
    feature_df: pd.DataFrame,
    config: Any | None = None,
) -> tuple[Path, Path]:
    """Save the feature table and feature column list to disk.

    Parameters
    ----------
    feature_df : pd.DataFrame
        The complete feature table.
    config : dict, optional
        Configuration dictionary. If None, loads from config.yaml.

    Returns
    -------
    tuple[Path, Path]
        Paths to the saved parquet file and feature columns JSON file.
    """
    config_data = _load_config_for_builder(config)
    processed_dir = config_data.get("data_paths", {}).get("processed_dir", "data/processed")
    artifacts_dir = config_data.get("data_paths", {}).get("artifacts_dir", "outputs/artifacts")

    repo_root = Path(__file__).resolve().parents[3]

    processed_path = Path(processed_dir)
    if not processed_path.is_absolute():
        processed_path = repo_root / processed_path
    processed_path.mkdir(parents=True, exist_ok=True)
    parquet_path = processed_path / "features.parquet"

    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.is_absolute():
        artifacts_path = repo_root / artifacts_path
    artifacts_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving feature table to %s", parquet_path)
    feature_df.to_parquet(parquet_path, index=False)

    identifier_cols = {
        "date",
        "home_team",
        "away_team",
        "tournament",
        "neutral",
        "home_score",
        "away_score",
        "goal_diff",
        "outcome",
    }

    feature_cols = [col for col in feature_df.columns if col not in identifier_cols]

    logger.info("Saving feature columns list (%d columns) to %s", len(feature_cols), artifacts_path)
    save_artifact(feature_cols, "feature_columns.json", artifacts_path)

    return parquet_path, artifacts_path / "feature_columns.json"
