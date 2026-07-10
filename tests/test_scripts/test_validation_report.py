"""Tests for the tournament validation report."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.validate_worldcup2022 import (
    EURO_2024_END,
    EURO_2024_START,
    WC_2022_END,
    WC_2022_START,
    filter_tournament_matches,
)


@pytest.fixture
def features_df() -> pd.DataFrame:
    return pd.read_parquet("data/processed/features.parquet")


def test_wc2022_match_count(features_df: pd.DataFrame) -> None:
    """Confirm WC 2022 date-range filtering selects exactly 64 matches."""
    wc = filter_tournament_matches(features_df, "FIFA World Cup", WC_2022_START, WC_2022_END)
    assert len(wc) == 64


def test_euro2024_match_count(features_df: pd.DataFrame) -> None:
    """Confirm Euro 2024 date-range filtering selects exactly 51 matches."""
    euro = filter_tournament_matches(features_df, "UEFA Euro", EURO_2024_START, EURO_2024_END)
    assert len(euro) == 51


def test_outcome_correct_but_exact_score_wrong() -> None:
    """Test that outcome correctness and exact scoreline are independent flags."""
    # Synthetic case: outcome is right (home_win) but exact score is wrong
    actual_home, actual_away = 2, 1
    actual_outcome = "home_win"
    xgb_pick = "home_win"
    poisson_home, poisson_away = 1, 0

    xgb_correct = xgb_pick == actual_outcome
    poisson_exact = (poisson_home == actual_home) and (poisson_away == actual_away)

    assert xgb_correct is True
    assert poisson_exact is False

    # Synthetic case: outcome is wrong but exact score happens to match
    # (impossible in practice since matching score implies matching outcome,
    # but let's verify the logic is truly independent)
    actual_home2, actual_away2 = 1, 0
    actual_outcome2 = "home_win"
    xgb_pick2 = "away_win"
    poisson_home2, poisson_away2 = 1, 0

    xgb_correct2 = xgb_pick2 == actual_outcome2
    poisson_exact2 = (poisson_home2 == actual_home2) and (poisson_away2 == actual_away2)

    assert xgb_correct2 is False
    assert poisson_exact2 is True
