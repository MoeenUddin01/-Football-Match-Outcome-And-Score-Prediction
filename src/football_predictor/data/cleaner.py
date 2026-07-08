"""Clean and deduplicate raw international football match results."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def clean_results(results: pd.DataFrame) -> pd.DataFrame:
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
    return cleaned
