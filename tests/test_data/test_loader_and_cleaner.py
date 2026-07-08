from pathlib import Path

import pandas as pd

from src.football_predictor.data.cleaner import clean_results
from src.football_predictor.data.loader import (
    load_former_names,
    load_goalscorers,
    load_results,
    load_shootouts,
)


def test_loader_functions_read_and_parse_dates(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "home_score": [1, 2],
            "away_score": [0, 1],
            "tournament": ["Friendly", "World Cup"],
            "city": ["x", "y"],
            "country": ["z", "w"],
            "neutral": [False, True],
        }
    ).to_csv(data_dir / "results.csv", index=False)

    pd.DataFrame(
        {
            "date": ["2023-01-01"],
            "home_team": ["A"],
            "away_team": ["B"],
            "team": ["A"],
            "scorer": ["X"],
            "minute": [10],
            "own_goal": [False],
            "penalty": [False],
        }
    ).to_csv(data_dir / "goalscorers.csv", index=False)

    pd.DataFrame(
        {
            "date": ["2023-01-01"],
            "home_team": ["A"],
            "away_team": ["B"],
            "winner": ["A"],
            "first_shooter": ["A"],
        }
    ).to_csv(data_dir / "shootouts.csv", index=False)

    pd.DataFrame(
        {
            "current": ["A"],
            "former": ["Alpha"],
            "start_date": ["2020-01-01"],
            "end_date": ["2021-01-01"],
        }
    ).to_csv(data_dir / "former_names.csv", index=False)

    results = load_results(data_dir)
    goalscorers = load_goalscorers(data_dir)
    shootouts = load_shootouts(data_dir)
    former_names = load_former_names(data_dir)

    assert isinstance(results, pd.DataFrame)
    assert isinstance(goalscorers, pd.DataFrame)
    assert isinstance(shootouts, pd.DataFrame)
    assert isinstance(former_names, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(results["date"])
    assert pd.api.types.is_datetime64_any_dtype(goalscorers["date"])
    assert pd.api.types.is_datetime64_any_dtype(shootouts["date"])
    assert pd.api.types.is_datetime64_any_dtype(former_names["start_date"])
    assert pd.api.types.is_datetime64_any_dtype(former_names["end_date"])


def test_clean_results_handles_duplicates_and_null_scores(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
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
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "x",
                "country": "y",
                "neutral": False,
            },
            {
                "date": "2023-01-03",
                "home_team": "A",
                "away_team": "B",
                "home_score": None,
                "away_score": None,
                "tournament": "Friendly",
                "city": "x",
                "country": "y",
                "neutral": False,
            },
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
    ).to_csv(data_dir / "results.csv", index=False)

    results = load_results(data_dir)
    cleaned = clean_results(results)

    assert len(cleaned) == 2
    assert cleaned["date"].is_monotonic_increasing
    assert cleaned.iloc[0]["date"].strftime("%Y-%m-%d") == "2023-01-01"
    assert cleaned.iloc[1]["date"].strftime("%Y-%m-%d") == "2023-01-02"
    assert set(cleaned["home_team"]) == {"A"}
