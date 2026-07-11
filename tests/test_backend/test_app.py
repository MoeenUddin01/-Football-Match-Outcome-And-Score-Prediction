"""Tests for the FastAPI backend endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import app, load_models


@pytest.fixture(scope="module", autouse=True)
def _load_startup() -> None:
    """Trigger model loading once for all tests."""
    load_models()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_get_teams(client: TestClient) -> None:
    resp = client.get("/api/teams")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "Brazil" in data
    assert "Argentina" in data


def test_get_rankings(client: TestClient) -> None:
    resp = client.get("/api/rankings")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    assert "rank" in first
    assert "team" in first
    assert "elo" in first
    assert first["rank"] == 1
    assert data[0]["elo"] >= data[1]["elo"]


def test_get_tournaments(client: TestClient) -> None:
    resp = client.get("/api/tournaments")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    names = {t["name"] for t in data}
    assert "FIFA World Cup" in names
    assert "Friendly" in names
    for t in data:
        assert "name" in t
        assert "tier" in t


def test_post_predict_success(client: TestClient) -> None:
    resp = client.post("/api/predict", json={
        "home_team": "Brazil",
        "away_team": "Argentina",
        "tournament": "FIFA World Cup",
        "neutral": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["home_team"] == "Brazil"
    assert data["away_team"] == "Argentina"
    assert "xgb_probs" in data
    assert "home_win" in data["xgb_probs"]
    assert "draw" in data["xgb_probs"]
    assert "away_win" in data["xgb_probs"]
    assert "xgb_pick" in data
    assert "poisson_home_goals" in data
    assert "poisson_away_goals" in data
    assert "poisson_probs" in data


def test_post_predict_unknown_team(client: TestClient) -> None:
    resp = client.post("/api/predict", json={
        "home_team": "Narnia",
        "away_team": "Brazil",
        "tournament": "Friendly",
        "neutral": False,
    })
    assert resp.status_code == 404
    data = resp.json()
    assert "detail" in data
    assert "Narnia" in data["detail"]


def test_validation_report(client: TestClient) -> None:
    resp = client.get("/api/validation-report")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 115
    first = data[0]
    for key in ("tournament", "date", "home_team", "away_team",
                "home_score", "away_score", "actual_outcome",
                "xgb_probs", "xgb_pick", "xgb_correct",
                "poisson_home_goals", "poisson_away_goals", "poisson_exact"):
        assert key in first


# ---------- Top-scorers endpoint tests ----------


def test_top_scorers_returns_results(client: TestClient) -> None:
    resp = client.get("/api/top-scorers")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    assert "scorer" in first
    assert "team" in first
    assert "goal_count" in first
    assert "penalty_count" in first
    assert "own_goal_count" in first
    # Should be sorted by goal_count descending
    assert data[0]["goal_count"] >= data[1]["goal_count"]


def test_top_scorers_limit(client: TestClient) -> None:
    resp = client.get("/api/top-scorers?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 5


def test_top_scorers_team_filter(client: TestClient) -> None:
    resp = client.get("/api/top-scorers?team=Brazil")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) > 0
    for row in data:
        assert row["team"] == "Brazil"


def test_top_scorers_tournament_filter(client: TestClient) -> None:
    resp = client.get("/api/top-scorers?tournament=FIFA World Cup&limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3


def test_top_scorers_null_scorers_excluded(client: TestClient) -> None:
    resp = client.get("/api/top-scorers?limit=500")
    data = resp.json()
    for row in data:
        assert row["scorer"] is not None
        assert row["scorer"] != "null"
        assert row["scorer"] != "None"


def test_top_scorers_own_goals_excluded(client: TestClient) -> None:
    """Own goals must NOT count toward a player's goal_count.

    An own goal is scored against the scorer's team, not credited to
    them.  This test constructs a synthetic scenario to verify the
    exclusion logic.
    """
    import pandas as pd
    from backend.app import _compute_top_scorers

    synthetic = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 4),
        "home_team": ["A", "A", "A", "A"],
        "away_team": ["B", "B", "B", "B"],
        "team": ["A", "A", "A", "A"],
        "scorer": ["Alice", "Alice", "Alice", "Alice"],
        "minute": [10, 20, 30, 40],
        "own_goal": [False, False, False, True],   # 1 own goal
        "penalty": [False, False, False, False],
        "tournament": ["Test"] * 4,
    })

    result = _compute_top_scorers(synthetic, limit=10)
    alice = result[0]
    # Alice has 3 real goals + 1 own goal. Own goal should NOT count.
    assert alice["goal_count"] == 3
    assert alice["own_goal_count"] == 1


def test_top_scorers_team_endpoint(client: TestClient) -> None:
    resp = client.get("/api/top-scorers/team/Brazil")
    assert resp.status_code == 200
    data = resp.json()
    assert data["team"] == "Brazil"
    assert "total_goals" in data
    assert data["total_goals"] > 0
    assert "scorers" in data
    assert len(data["scorers"]) <= 10
    for s in data["scorers"]:
        assert "scorer" in s
        assert "goal_count" in s


def test_top_scorers_team_not_found(client: TestClient) -> None:
    resp = client.get("/api/top-scorers/team/Narnia")
    assert resp.status_code == 404
