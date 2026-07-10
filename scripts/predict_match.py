"""Predict the outcome and scoreline of a single football match.

Usage:
    python scripts/predict_match.py --home "Brazil" --away "Argentina" \
        --tournament "FIFA World Cup" --neutral
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from football_predictor.models.outcome_classifier import _build_feature_matrix
from football_predictor.models.score_regressor import poisson_to_outcome_probs
from football_predictor.utils.io import load_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "outputs" / "artifacts"
MODELS_DIR = REPO_ROOT / "outputs" / "models"


def _load_artifacts() -> dict:
    """Load all persisted model artifacts."""
    return {
        "elo": load_artifact("team_elo_latest.json", ARTIFACTS_DIR),
        "tmap": load_artifact("tournament_tier_map.json", ARTIFACTS_DIR),
        "final_feature_columns": load_artifact("final_feature_columns.json", ARTIFACTS_DIR),
        "encoder": load_artifact("tournament_tier_encoder.pkl", ARTIFACTS_DIR),
        "clf": load_artifact("outcome_classifier.pkl", MODELS_DIR),
        "home_model": load_artifact("home_score_regressor.pkl", MODELS_DIR),
        "away_model": load_artifact("away_score_regressor.pkl", MODELS_DIR),
        "scaler": load_artifact("score_regressor_scaler.pkl", ARTIFACTS_DIR),
    }


def _get_team_elo(team: str, elo_dict: dict) -> float | None:
    """Get latest Elo rating for a team."""
    return elo_dict.get(team)


def _get_team_form(team: str, side: str, features_df: pd.DataFrame) -> dict:
    """Get the most recent rolling stats for a team.

    Parameters
    ----------
    team : str
        Team name.
    side : str
        'home' or 'away' — which columns to read.
    features_df : pd.DataFrame
        The full features table.

    Returns
    -------
    dict
        Latest rolling feature values for this team.
    """
    col_prefix = f"{side}_"
    mask = features_df[f"{side}_team"] == team
    if not mask.any():
        return {}
    recent = features_df.loc[mask].sort_values("date").iloc[-1]
    result = {}
    for col in features_df.columns:
        if col.startswith(col_prefix) and col not in (
            f"{side}_team", f"{side}_score", f"{side}_elo_post",
        ):
            result[col] = recent[col]
    return result


def _get_h2h(home_team: str, away_team: str, features_df: pd.DataFrame) -> dict:
    """Get the most recent head-to-head record between two teams."""
    mask = (
        ((features_df["home_team"] == home_team) & (features_df["away_team"] == away_team))
        | ((features_df["home_team"] == away_team) & (features_df["away_team"] == home_team))
    )
    if not mask.any():
        return {"h2h_home_wins": 0, "h2h_away_wins": 0, "h2h_draws": 0}
    recent = features_df.loc[mask].sort_values("date").iloc[-1]
    return {
        "h2h_home_wins": recent["h2h_home_wins"],
        "h2h_away_wins": recent["h2h_away_wins"],
        "h2h_draws": recent["h2h_draws"],
    }


def _resolve_tier(tournament: str, tmap: dict) -> str:
    """Map tournament name to tier string."""
    return tmap.get(tournament, "default")


def predict_match(
    home_team: str,
    away_team: str,
    tournament: str,
    neutral: bool,
    artifacts: dict,
    features_df: pd.DataFrame,
) -> dict | None:
    """Build feature vector and run predictions.

    Returns None if either team has no historical data.
    """
    elo = artifacts["elo"]
    tmap = artifacts["tmap"]
    final_feature_columns = artifacts["final_feature_columns"]
    encoder = artifacts["encoder"]
    clf = artifacts["clf"]
    home_model = artifacts["home_model"]
    away_model = artifacts["away_model"]
    scaler = artifacts["scaler"]

    # Check both teams exist in Elo
    home_elo = _get_team_elo(home_team, elo)
    away_elo = _get_team_elo(away_team, elo)
    if home_elo is None or away_elo is None:
        return None

    # Build feature dict
    feature = {}
    feature["home_elo_pre"] = home_elo
    feature["away_elo_pre"] = away_elo
    feature["elo_diff"] = home_elo - away_elo

    # Rolling form
    home_form = _get_team_form(home_team, "home", features_df)
    away_form = _get_team_form(away_team, "away", features_df)
    feature.update(home_form)
    feature.update(away_form)

    # If home team was never home before, try away side stats (for new matchups)
    if not home_form:
        home_form_alt = _get_team_form(home_team, "away", features_df)
        for k, v in home_form_alt.items():
            alt_key = k.replace("away_", "home_")
            feature[alt_key] = v
    if not away_form:
        away_form_alt = _get_team_form(away_team, "home", features_df)
        for k, v in away_form_alt.items():
            alt_key = k.replace("home_", "away_")
            feature[alt_key] = v

    # H2H
    h2h = _get_h2h(home_team, away_team, features_df)
    feature.update(h2h)

    # Tournament tier
    tier = _resolve_tier(tournament, tmap)
    feature["tournament_tier"] = tier

    # Neutral venue
    feature["neutral"] = int(neutral)

    # Build DataFrame with correct column order
    numeric_cols = [c for c in final_feature_columns if not c.startswith("tournament_tier_")]
    feature_row = {c: feature.get(c, 0.0) for c in numeric_cols}
    feature_row["tournament_tier"] = tier

    match_df = pd.DataFrame([feature_row])
    for col in numeric_cols:
        match_df[col] = pd.to_numeric(match_df[col], errors="coerce")

    X = _build_feature_matrix(match_df, encoder, numeric_cols, final_feature_columns)
    X_scaled = scaler.transform(X)

    # Classifier
    probs_raw = clf.predict_proba(X)
    xgb_probs = {
        "home_win": float(probs_raw[0][2]),
        "draw": float(probs_raw[0][1]),
        "away_win": float(probs_raw[0][0]),
    }
    pick = max(xgb_probs, key=xgb_probs.get)

    # Regressor
    home_lambda = max(float(home_model.predict(X_scaled)[0]), 0.01)
    away_lambda = max(float(away_model.predict(X_scaled)[0]), 0.01)
    home_goals = int(round(home_lambda))
    away_goals = int(round(away_lambda))

    # Poisson-derived outcome probs
    poisson_probs = poisson_to_outcome_probs(home_lambda, away_lambda)

    return {
        "xgb_probs": xgb_probs,
        "xgb_pick": pick,
        "poisson_home_lambda": home_lambda,
        "poisson_away_lambda": away_lambda,
        "poisson_home_goals": home_goals,
        "poisson_away_goals": away_goals,
        "poisson_probs": poisson_probs,
    }


def format_output(
    home_team: str,
    away_team: str,
    tournament: str,
    neutral: bool,
    result: dict,
) -> str:
    """Format prediction result as readable text."""
    venue = "neutral venue" if neutral else "home venue"
    lines = [
        f"{home_team} vs {away_team} ({tournament}, {venue})",
        "",
        "XGBoost classifier:",
        f"  {home_team} {result['xgb_probs']['home_win']:.0%} | "
        f"Draw {result['xgb_probs']['draw']:.0%} | "
        f"{away_team} {result['xgb_probs']['away_win']:.0%}",
        f"  Pick: {result['xgb_pick']}",
        "",
        "Poisson scoreline:",
        f"  {home_team} {result['poisson_home_goals']} - {result['poisson_away_goals']} {away_team}",
        f"  (expected goals: {result['poisson_home_lambda']:.2f} - {result['poisson_away_lambda']:.2f})",
        "",
        "Poisson-derived outcome probs:",
        f"  {home_team} {result['poisson_probs']['home_win_prob']:.0%} | "
        f"Draw {result['poisson_probs']['draw_prob']:.0%} | "
        f"{away_team} {result['poisson_probs']['away_win_prob']:.0%}",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Predict a football match outcome and scoreline.")
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument("--tournament", required=True, help="Tournament name (e.g. 'FIFA World Cup')")
    parser.add_argument("--neutral", action="store_true", default=False, help="Neutral venue flag")
    args = parser.parse_args()

    features_df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features.parquet")
    artifacts = _load_artifacts()

    result = predict_match(args.home, args.away, args.tournament, args.neutral, artifacts, features_df)

    if result is None:
        missing = []
        if _get_team_elo(args.home, artifacts["elo"]) is None:
            missing.append(args.home)
        if _get_team_elo(args.away, artifacts["elo"]) is None:
            missing.append(args.away)
        print(f"Cannot predict: {', '.join(missing)} not found in historical data.")
        print("These teams have no Elo rating or match history in the dataset.")
        sys.exit(1)

    print(format_output(args.home, args.away, args.tournament, args.neutral, result))


if __name__ == "__main__":
    main()
