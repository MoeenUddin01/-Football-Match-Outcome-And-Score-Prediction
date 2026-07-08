from pathlib import Path

import pandas as pd

from src.football_predictor.data.loader import load_results


def test_load_results_reads_and_parses_dates(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "date": ["2023-01-01"],
            "home_team": ["A"],
            "away_team": ["B"],
            "home_score": [1],
            "away_score": [0],
            "tournament": ["Friendly"],
            "city": ["x"],
            "country": ["y"],
            "neutral": [False],
        }
    ).to_csv(data_dir / "results.csv", index=False)

    results = load_results(data_dir)

    assert len(results) == 1
    assert pd.api.types.is_datetime64_any_dtype(results["date"])
    assert results.iloc[0]["home_team"] == "A"
