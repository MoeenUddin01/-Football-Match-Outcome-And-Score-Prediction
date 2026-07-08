import pandas as pd

from src.football_predictor.data.cleaner import clean_results


def test_clean_results_drops_null_scores_and_resolves_duplicates() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2023-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "x",
                "country": "y",
                "neutral": False,
            },
            {
                "date": "2023-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "x",
                "country": "y",
                "neutral": False,
            },
            {
                "date": "2023-01-02",
                "home_team": "A",
                "away_team": "B",
                "home_score": None,
                "away_score": None,
                "tournament": "Friendly",
                "city": "x",
                "country": "y",
                "neutral": False,
            },
        ]
    )

    cleaned = clean_results(frame)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["home_team"] == "A"


def test_clean_results_excludes_conflicting_duplicates() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2023-01-04",
                "home_team": "C",
                "away_team": "D",
                "home_score": 1,
                "away_score": 2,
                "tournament": "Friendly",
                "city": "x",
                "country": "y",
                "neutral": False,
            },
            {
                "date": "2023-01-04",
                "home_team": "C",
                "away_team": "D",
                "home_score": 0,
                "away_score": 2,
                "tournament": "Friendly",
                "city": "x",
                "country": "y",
                "neutral": False,
            },
        ]
    )

    cleaned = clean_results(frame)

    assert cleaned.empty
