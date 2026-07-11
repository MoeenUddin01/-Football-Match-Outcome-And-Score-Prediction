"""Load raw football datasets from the local CSV files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _read_csv(path: str | Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    data_path = Path(path)
    dataframe = pd.read_csv(data_path, parse_dates=parse_dates or [])
    logger.info("Loaded %s rows from %s", len(dataframe), data_path)
    return dataframe


def load_results(data_dir: str | Path) -> pd.DataFrame:
    """Load the results dataset and enforce expected dtypes."""
    data_path = Path(data_dir) / "results.csv"
    dataframe = _read_csv(data_path, parse_dates=["date"])
    dataframe = dataframe.copy()
    dataframe["date"] = pd.to_datetime(dataframe["date"], utc=False)
    dataframe["home_team"] = dataframe["home_team"].astype(str)
    dataframe["away_team"] = dataframe["away_team"].astype(str)
    dataframe["tournament"] = dataframe["tournament"].astype(str)
    dataframe["city"] = dataframe["city"].astype(str)
    dataframe["country"] = dataframe["country"].astype(str)
    dataframe["neutral"] = dataframe["neutral"].fillna(False).astype(bool)
    dataframe["home_score"] = pd.to_numeric(dataframe["home_score"], errors="coerce")
    dataframe["away_score"] = pd.to_numeric(dataframe["away_score"], errors="coerce")
    return dataframe


def load_goalscorers(data_dir: str | Path) -> pd.DataFrame:
    """Load goalscorer data and enforce expected dtypes."""
    data_path = Path(data_dir) / "goalscorers.csv"
    dataframe = _read_csv(data_path, parse_dates=["date"])
    dataframe = dataframe.copy()
    dataframe["date"] = pd.to_datetime(dataframe["date"], utc=False)
    dataframe["home_team"] = dataframe["home_team"].astype(str)
    dataframe["away_team"] = dataframe["away_team"].astype(str)
    dataframe["team"] = dataframe["team"].astype(str)
    dataframe["scorer"] = dataframe["scorer"].astype("string")
    dataframe["minute"] = pd.to_numeric(dataframe["minute"], errors="coerce")
    dataframe["own_goal"] = dataframe["own_goal"].fillna(False).astype(bool)
    dataframe["penalty"] = dataframe["penalty"].fillna(False).astype(bool)
    return dataframe


def load_shootouts(data_dir: str | Path) -> pd.DataFrame:
    """Load shootout data and enforce expected dtypes."""
    data_path = Path(data_dir) / "shootouts.csv"
    dataframe = _read_csv(data_path, parse_dates=["date"])
    dataframe = dataframe.copy()
    dataframe["date"] = pd.to_datetime(dataframe["date"], utc=False)
    dataframe["home_team"] = dataframe["home_team"].astype(str)
    dataframe["away_team"] = dataframe["away_team"].astype(str)
    dataframe["winner"] = dataframe["winner"].astype(str)
    dataframe["first_shooter"] = dataframe["first_shooter"].astype("string")
    return dataframe


def load_former_names(data_dir: str | Path) -> pd.DataFrame:
    """Load historical team-name mapping data and enforce expected dtypes."""
    data_path = Path(data_dir) / "former_names.csv"
    dataframe = _read_csv(data_path, parse_dates=["start_date", "end_date"])
    dataframe = dataframe.copy()
    dataframe["start_date"] = pd.to_datetime(dataframe["start_date"], utc=False)
    dataframe["end_date"] = pd.to_datetime(dataframe["end_date"], utc=False)
    dataframe["current"] = dataframe["current"].astype(str)
    dataframe["former"] = dataframe["former"].astype(str)
    return dataframe
