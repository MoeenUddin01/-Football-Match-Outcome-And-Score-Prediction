from pathlib import Path

from src.football_predictor.utils.io import load_artifact, save_artifact


def test_save_and_load_json_artifact(tmp_path: Path) -> None:
    payload = {"team": "Brazil", "rating": 1800}

    path = save_artifact(payload, "team_rating", tmp_path)

    assert path.exists()
    assert load_artifact("team_rating", tmp_path) == payload


def test_save_and_load_pickle_artifact(tmp_path: Path) -> None:
    payload = ["a", "b", 3]

    path = save_artifact(payload, "list_artifact", tmp_path)

    assert path.exists()
    assert load_artifact("list_artifact", tmp_path) == payload
