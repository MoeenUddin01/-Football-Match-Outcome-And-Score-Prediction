"""Clean and deduplicate raw international football match results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _default_output_path() -> Path:
    """Return the default path for the cleaned parquet file from config."""
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config" / "config.yaml"
    if not config_path.exists():
        return repo_root / "data" / "processed" / "results_cleaned.parquet"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    processed_dir = config.get("data_paths", {}).get("processed_dir", "data/processed")
    processed_path = Path(processed_dir)
    if not processed_path.is_absolute():
        processed_path = repo_root / processed_path
    processed_path.mkdir(parents=True, exist_ok=True)
    return processed_path / "results_cleaned.parquet"


def save_cleaned_results(results: pd.DataFrame, output_path: str | Path | None = None) -> Path:
    """Persist a cleaned results dataframe as a parquet file."""
    target_path = Path(output_path) if output_path is not None else _default_output_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(target_path, index=False)
    logger.info("Saved cleaned results parquet to %s", target_path)
    return target_path


def clean_results(results: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    """Clean the results dataset by removing null-score rows and resolving duplicates.

    The function performs the following steps in order:
    1. Drop rows with missing home/away scores, logging the number removed.
    2. Sort the remaining rows by date ascending.
    3. Resolve duplicate (date, home_team, away_team) groups by keeping one row
       if all scores match exactly, or excluding all conflicting rows if scores
       differ within the group.
    4. Log a summary of the cleaning outcomes.
    """
    if results.empty:
        logger.info("clean_results: input had no rows; returning empty dataframe")
        return results.copy()

    frame = results.copy()
    initial_rows = len(frame)

    frame = frame.dropna(subset=["home_score", "away_score"]).copy()
    null_score_rows_dropped = initial_rows - len(frame)
    logger.info(
        "clean_results: dropped %s row(s) with null scores",
        null_score_rows_dropped,
    )

    frame = frame.sort_values("date").reset_index(drop=True)

    duplicate_groups = frame.groupby(["date", "home_team", "away_team"], dropna=False)
    resolved_duplicate_count = 0
    excluded_conflict_count = 0
    kept_rows: list[pd.Series] = []

    for _, group in duplicate_groups:
        if len(group) == 1:
            kept_rows.append(group.iloc[0])
            continue

        group_scores = set(
            (row["home_score"], row["away_score"]) for _, row in group.iterrows()
        )
        if len(group_scores) == 1:
            resolved_duplicate_count += len(group) - 1
            kept_rows.append(group.iloc[0])
            logger.info(
                "resolved duplicate: %s / %s on %s between %s and %s",
                group.iloc[0]["home_score"],
                group.iloc[0]["away_score"],
                group.iloc[0]["date"],
                group.iloc[0]["home_team"],
                group.iloc[0]["away_team"],
            )
        else:
            excluded_conflict_count += len(group)
            conflicting_scores = sorted(group_scores)
            logger.warning(
                "unresolved conflict — excluded group for %s between %s and %s with scores %s",
                group.iloc[0]["date"],
                group.iloc[0]["home_team"],
                group.iloc[0]["away_team"],
                conflicting_scores,
            )

    cleaned = pd.DataFrame(kept_rows).reset_index(drop=True)
    logger.info(
        "clean_results summary: rows in=%s, dropped null scores=%s, duplicates resolved=%s, duplicates excluded as conflicts=%s, final rows=%s",
        initial_rows,
        null_score_rows_dropped,
        resolved_duplicate_count,
        excluded_conflict_count,
        len(cleaned),
    )

    if output_path is None:
        save_cleaned_results(cleaned)
    else:
        save_cleaned_results(cleaned, output_path=output_path)

    return cleaned
