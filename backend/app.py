"""FastAPI backend serving the football prediction models as a REST API."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure the project root is on sys.path so we can import from src/
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.predict_match import (
    _load_artifacts,
    _get_team_elo,
    format_output,
    predict_match,
)
from football_predictor.utils.io import load_artifact

# ---------- State loaded once at startup ----------

ARTIFACTS: dict[str, Any] = {}
FEATURES_DF: pd.DataFrame = pd.DataFrame()


def load_models() -> None:
    """Load all trained models and data into module globals."""
    global ARTIFACTS, FEATURES_DF
    ARTIFACTS = _load_artifacts()
    FEATURES_DF = pd.read_parquet(_REPO_ROOT / "data" / "processed" / "features.parquet")


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_models()
    yield


app = FastAPI(title="Football Prediction API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request / response models ----------


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    tournament: str
    neutral: bool = False


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    tournament: str
    neutral: bool
    xgb_probs: dict[str, float]
    xgb_pick: str
    poisson_home_goals: int
    poisson_away_goals: int
    poisson_home_lambda: float
    poisson_away_lambda: float
    poisson_probs: dict[str, float]


# ---------- Endpoints ----------


@app.get("/api/teams")
def get_teams() -> list[str]:
    """Return every team name in the dataset, sorted alphabetically."""
    teams = sorted(
        set(FEATURES_DF["home_team"].unique()) | set(FEATURES_DF["away_team"].unique())
    )
    return teams


@app.get("/api/rankings")
def get_rankings() -> list[dict[str, Any]]:
    """Return teams ranked by current Elo, highest first."""
    elo = ARTIFACTS["elo"]
    ranked = sorted(elo.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {"rank": i + 1, "team": team, "elo": round(rating, 1)}
        for i, (team, rating) in enumerate(ranked)
    ]


@app.get("/api/tournaments")
def get_tournaments() -> list[dict[str, str]]:
    """Return distinct tournament names and their tiers."""
    tmap = ARTIFACTS["tmap"]
    return [{"name": k, "tier": v} for k, v in sorted(tmap.items())]


@app.post("/api/predict")
def predict(req: PredictRequest) -> PredictResponse:
    """Predict outcome and scoreline for a match."""
    result = predict_match(
        req.home_team,
        req.away_team,
        req.tournament,
        req.neutral,
        ARTIFACTS,
        FEATURES_DF,
    )

    if result is None:
        missing = []
        if _get_team_elo(req.home_team, ARTIFACTS["elo"]) is None:
            missing.append(req.home_team)
        if _get_team_elo(req.away_team, ARTIFACTS["elo"]) is None:
            missing.append(req.away_team)
        raise HTTPException(
            status_code=404,
            detail=f"Team(s) not found in historical data: {', '.join(missing)}",
        )

    return PredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        tournament=req.tournament,
        neutral=req.neutral,
        xgb_probs=result["xgb_probs"],
        xgb_pick=result["xgb_pick"],
        poisson_home_goals=result["poisson_home_goals"],
        poisson_away_goals=result["poisson_away_goals"],
        poisson_home_lambda=round(result["poisson_home_lambda"], 2),
        poisson_away_lambda=round(result["poisson_away_lambda"], 2),
        poisson_probs={k: round(v, 4) for k, v in result["poisson_probs"].items()},
    )


@app.get("/api/validation-report")
def get_validation_report() -> list[dict[str, Any]]:
    """Return WC 2022 + Euro 2024 backtest results as structured JSON."""
    from scripts.validate_worldcup2022 import (
        EURO_2024_END,
        EURO_2024_START,
        WC_2022_END,
        WC_2022_START,
        filter_tournament_matches,
        predict_single_match,
    )

    matches_out: list[dict[str, Any]] = []

    for tname, tconf in [
        ("FIFA World Cup", {"start": WC_2022_START, "end": WC_2022_END, "label": "FIFA World Cup 2022"}),
        ("UEFA Euro", {"start": EURO_2024_START, "end": EURO_2024_END, "label": "UEFA Euro 2024"}),
    ]:
        filtered = filter_tournament_matches(FEATURES_DF, tname, tconf["start"], tconf["end"])

        for _, match in filtered.iterrows():
            match_row = match.to_frame().T
            for col in match_row.columns:
                if col in ("date", "home_team", "away_team", "tournament", "outcome", "tournament_tier"):
                    continue
                match_row[col] = pd.to_numeric(match_row[col], errors="coerce")

            preds = predict_single_match(match_row, ARTIFACTS)

            actual_home = int(match["home_score"])
            actual_away = int(match["away_score"])
            actual_outcome = match["outcome"]
            xgb_correct = preds["xgb_pick"] == actual_outcome
            poisson_exact = (
                preds["poisson_home_goals"] == actual_home
                and preds["poisson_away_goals"] == actual_away
            )

            matches_out.append({
                "tournament": tconf["label"],
                "date": pd.Timestamp(match["date"]).strftime("%Y-%m-%d"),
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "home_score": actual_home,
                "away_score": actual_away,
                "actual_outcome": actual_outcome,
                "xgb_probs": preds["xgb_probs"],
                "xgb_pick": preds["xgb_pick"],
                "xgb_correct": xgb_correct,
                "poisson_home_goals": preds["poisson_home_goals"],
                "poisson_away_goals": preds["poisson_away_goals"],
                "poisson_exact": poisson_exact,
            })

    return matches_out
