"""Tests for the temporal splitter module."""

from __future__ import annotations

import pandas as pd
import pytest

from football_predictor.evaluation.splitter import split_temporal


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Create a sample features DataFrame spanning multiple years."""
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="QE")
    data = {
        "date": dates,
        "home_team": ["Team A"] * len(dates),
        "away_team": ["Team B"] * len(dates),
        "outcome": ["home_win"] * len(dates),
        "home_elo_pre": [1500] * len(dates),
        "away_elo_pre": [1500] * len(dates),
    }
    return pd.DataFrame(data)


def test_no_date_overlap_between_splits(sample_features_df: pd.DataFrame) -> None:
    """Confirm no date overlap between train/val/test splits."""
    config = {"date_splits": {"train_end": "2018-12-31", "val_end": "2022-12-31"}}
    train_df, val_df, test_df = split_temporal(sample_features_df, config=config)

    train_dates = set(train_df["date"])
    val_dates = set(val_df["date"])
    test_dates = set(test_df["date"])

    assert len(train_dates & val_dates) == 0, "Overlap between train and val"
    assert len(train_dates & test_dates) == 0, "Overlap between train and test"
    assert len(val_dates & test_dates) == 0, "Overlap between val and test"


def test_row_counts_sum_to_total(sample_features_df: pd.DataFrame) -> None:
    """Confirm row counts of all splits sum to the total."""
    config = {"date_splits": {"train_end": "2018-12-31", "val_end": "2022-12-31"}}
    train_df, val_df, test_df = split_temporal(sample_features_df, config=config)

    total_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_rows == len(sample_features_df)


def test_raises_error_on_empty_split() -> None:
    """Confirm error raised on empty split."""
    data = {
        "date": ["2010-01-01", "2010-06-01"],
        "home_team": ["A", "A"],
        "away_team": ["B", "B"],
        "outcome": ["home_win", "draw"],
    }
    df = pd.DataFrame(data)
    config = {"date_splits": {"train_end": "2020-12-31", "val_end": "2022-12-31"}}

    with pytest.raises(ValueError, match="empty"):
        split_temporal(df, config=config)


def test_correct_date_boundaries(sample_features_df: pd.DataFrame) -> None:
    """Confirm correct date boundaries for each split."""
    config = {"date_splits": {"train_end": "2018-12-31", "val_end": "2022-12-31"}}
    train_df, val_df, test_df = split_temporal(sample_features_df, config=config)

    assert train_df["date"].max() <= pd.Timestamp("2018-12-31")
    assert val_df["date"].min() > pd.Timestamp("2018-12-31")
    assert val_df["date"].max() <= pd.Timestamp("2022-12-31")
    assert test_df["date"].min() > pd.Timestamp("2022-12-31")
