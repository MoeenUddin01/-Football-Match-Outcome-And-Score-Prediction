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
