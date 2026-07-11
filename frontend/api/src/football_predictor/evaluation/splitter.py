"""Temporal train/validation/test split for walk-forward evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


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


def split_temporal(
    features_df: pd.DataFrame,
    config: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split features into train/val/test by date.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature table with a 'date' column.
    config : dict, optional
        Configuration dictionary. If None, loads from config.yaml.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)

    Raises
    ------
    ValueError
        If any split is empty.
    """
    if not isinstance(features_df, pd.DataFrame):
        raise TypeError("features_df must be a pandas DataFrame")

    if "date" not in features_df.columns:
        raise KeyError("features_df must contain a 'date' column")

    config_data = _load_config(config)
    date_splits = config_data.get("date_splits", {})

    train_end_str = date_splits.get("train_end")
    val_end_str = date_splits.get("val_end")

    if not train_end_str or not val_end_str:
        raise ValueError(
            "config must contain date_splits.train_end and date_splits.val_end"
        )

    train_end = pd.Timestamp(train_end_str)
    val_end = pd.Timestamp(val_end_str)

    df = features_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["date"] <= train_end].copy()
    val_df = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test_df = df[df["date"] > val_end].copy()

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            raise ValueError(f"{name} split is empty")

    logger.info(
        "Temporal split: train=%d rows (date <= %s), val=%d rows (%s < date <= %s), test=%d rows (date > %s)",
        len(train_df),
        train_end_str,
        len(val_df),
        train_end_str,
        val_end_str,
        len(test_df),
        val_end_str,
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        min_date = split_df["date"].min()
        max_date = split_df["date"].max()
        logger.info("  %s: %d rows, date range %s to %s", name, len(split_df), min_date, max_date)

    return train_df, val_df, test_df
