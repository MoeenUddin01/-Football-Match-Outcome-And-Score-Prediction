"""Vercel serverless function — FastAPI backend wrapped with Mangum."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum

# ---------- Paths ----------
_API_DIR = Path(__file__).resolve().parent
_SRC = _API_DIR / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_predictor.models.outcome_classifier import _build_feature_matrix
from football_predictor.models.score_regressor import poisson_to_outcome_probs
from football_predictor.utils.io import load_artifact
from football_predictor.data.loader import load_goalscorers, load_results

# ---------- Constants ----------
ARTIFACTS_DIR = _API_DIR / "artifacts"
MODELS_DIR = _API_DIR / "models"
DATA_RAW = _API_DIR / "data" / "raw"
DATA_PROCESSED = _API_DIR / "data" / "processed"

# ---------- State (persists across warm invocations) ----------
ARTIFACTS: dict[str, Any] = {}
FEATURES_DF: pd.DataFrame = pd.DataFrame()
GOALSCORERS_DF: pd.DataFrame = pd.DataFrame()
_MODELS_LOADED = False


def _ensure_loaded() -> None:
    global ARTIFACTS, FEATURES_DF, GOALSCORERS_DF, _MODELS_LOADED
    if _MODELS_LOADED:
        return
    ARTIFACTS = {
        "elo": load_artifact("team_elo_latest.json", ARTIFACTS_DIR),
        "tmap": load_artifact("tournament_tier_map.json", ARTIFACTS_DIR),
        "final_feature_columns": load_artifact("final_feature_columns.json", ARTIFACTS_DIR),
        "encoder": load_artifact("tournament_tier_encoder.pkl", ARTIFACTS_DIR),
        "clf": load_artifact("outcome_classifier.pkl", MODELS_DIR),
        "home_model": load_artifact("home_score_regressor.pkl", MODELS_DIR),
        "away_model": load_artifact("away_score_regressor.pkl", MODELS_DIR),
        "scaler": load_artifact("score_regressor_scaler.pkl", ARTIFACTS_DIR),
    }
    FEATURES_DF = pd.read_csv(DATA_PROCESSED / "features.csv", parse_dates=["date"])

    scorers = load_goalscorers(DATA_RAW)
    results = load_results(DATA_RAW)
    results_tournament = results[["date", "home_team", "away_team", "tournament"]].drop_duplicates(
        subset=["date", "home_team", "away_team"]
    )
    GOALSCORERS_DF = scorers.merge(
        results_tournament, on=["date", "home_team", "away_team"], how="left"
    )
    _MODELS_LOADED = True


# ---------- FastAPI app ----------
app = FastAPI(title="Football Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
def health_check() -> dict[str, str]:
    _ensure_loaded()
    return {"status": "ok"}


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


# ---------- Helpers ----------


def _compute_top_scorers(
    df: pd.DataFrame,
    *,
    team: str | None = None,
    tournament: str | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    filtered = df.copy()
    if team:
        filtered = filtered[filtered["team"] == team]
    if tournament:
        filtered = filtered[filtered["tournament"] == tournament]

    filtered = filtered[filtered["scorer"].notna()]

    own_goal_counts = (
        filtered[filtered["own_goal"]]
        .groupby("scorer", sort=False)
        .size()
        .rename("own_goal_count")
    )

    non_own = filtered[~filtered["own_goal"]]

    agg = (
        non_own.groupby("scorer", sort=False)
        .agg(
            team=("team", "first"),
            goal_count=("scorer", "size"),
            penalty_count=("penalty", "sum"),
        )
        .reset_index()
    )

    agg = agg.merge(own_goal_counts, on="scorer", how="left")
    agg["own_goal_count"] = agg["own_goal_count"].fillna(0).astype(int)
    agg = agg.sort_values("goal_count", ascending=False).head(limit)

    return agg.to_dict(orient="records")


# ---------- Endpoints ----------


@app.get("/api/teams")
def get_teams() -> list[str]:
    _ensure_loaded()
    teams = sorted(
        set(FEATURES_DF["home_team"].unique()) | set(FEATURES_DF["away_team"].unique())
    )
    return teams


@app.get("/api/rankings")
def get_rankings() -> list[dict[str, Any]]:
    _ensure_loaded()
    elo = ARTIFACTS["elo"]
    ranked = sorted(elo.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {"rank": i + 1, "team": team, "elo": round(rating, 1)}
        for i, (team, rating) in enumerate(ranked)
    ]


@app.get("/api/tournaments")
def get_tournaments() -> list[dict[str, str]]:
    _ensure_loaded()
    tmap = ARTIFACTS["tmap"]
    return [{"name": k, "tier": v} for k, v in sorted(tmap.items())]


@app.post("/api/predict")
def predict(req: PredictRequest) -> PredictResponse:
    _ensure_loaded()

    # Import predict_match from the scripts module in api/
    scripts_dir = _API_DIR / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from predict_match import _get_team_elo, predict_match as _predict_match

    result = _predict_match(
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


@app.get("/api/top-scorers")
def get_top_scorers(
    team: str | None = Query(None),
    tournament: str | None = Query(None),
    limit: int = Query(25, ge=1, le=500),
) -> list[dict[str, Any]]:
    _ensure_loaded()
    return _compute_top_scorers(GOALSCORERS_DF, team=team, tournament=tournament, limit=limit)


@app.get("/api/top-scorers/team/{team_name}")
def get_top_scorers_team(team_name: str) -> dict[str, Any]:
    _ensure_loaded()
    team_df = GOALSCORERS_DF[GOALSCORERS_DF["team"] == team_name]

    if team_df.empty:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found in goalscorers data")

    valid = team_df[team_df["scorer"].notna()]

    own_goal_counts = (
        valid[valid["own_goal"]]
        .groupby("scorer", sort=False)
        .size()
        .rename("own_goal_count")
    )

    non_own = valid[~valid["own_goal"]]

    agg = (
        non_own.groupby("scorer", sort=False)
        .agg(
            team=("team", "first"),
            goal_count=("scorer", "size"),
            penalty_count=("penalty", "sum"),
        )
        .reset_index()
    )

    agg = agg.merge(own_goal_counts, on="scorer", how="left")
    agg["own_goal_count"] = agg["own_goal_count"].fillna(0).astype(int)
    agg = agg.sort_values("goal_count", ascending=False).head(10)

    total_goals = int(len(non_own))

    return {
        "team": team_name,
        "total_goals": total_goals,
        "scorers": agg.to_dict(orient="records"),
    }


@app.get("/api/validation-report")
def get_validation_report() -> list[dict[str, Any]]:
    _ensure_loaded()

    scripts_dir = _API_DIR / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from validate_worldcup2022 import (
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


# ---------- Vercel handler ----------
handler = Mangum(app)
