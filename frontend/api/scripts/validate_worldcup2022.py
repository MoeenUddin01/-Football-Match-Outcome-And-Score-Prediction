"""Validate trained models on World Cup 2022 and Euro 2024 matches.

Replays real tournament fixtures through the trained pipeline and
produces a human-readable report comparing predictions against actuals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from football_predictor.evaluation.metrics import (
    accuracy,
    brier_score,
    log_loss_score,
    predict_to_labels,
)
from football_predictor.models.outcome_classifier import _build_feature_matrix
from football_predictor.models.score_regressor import poisson_to_outcome_probs
from football_predictor.utils.io import load_artifact

logger = logging.getLogger(__name__)

# Real calendar dates sourced from FIFA and UEFA official schedules
# FIFA World Cup 2022: https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/qatar2022
WC_2022_START = "2022-11-20"
WC_2022_END = "2022-12-18"

# UEFA Euro 2024: https://www.uefa.com/uefaeuro/
EURO_2024_START = "2024-06-14"
EURO_2024_END = "2024-07-14"

TOURNAMENT_CONFIGS = [
    {
        "name": "FIFA World Cup 2022",
        "tournament": "FIFA World Cup",
        "start": WC_2022_START,
        "end": WC_2022_END,
    },
    {
        "name": "UEFA Euro 2024",
        "tournament": "UEFA Euro",
        "start": EURO_2024_START,
        "end": EURO_2024_END,
    },
]


def load_all_artifacts(artifacts_dir: str, models_dir: str) -> dict:
    """Load all trained model artifacts."""
    return {
        "encoder": load_artifact("tournament_tier_encoder.pkl", artifacts_dir),
        "final_feature_columns": load_artifact("final_feature_columns.json", artifacts_dir),
        "clf": load_artifact("outcome_classifier.pkl", models_dir),
        "home_model": load_artifact("home_score_regressor.pkl", models_dir),
        "away_model": load_artifact("away_score_regressor.pkl", models_dir),
        "scaler": load_artifact("score_regressor_scaler.pkl", artifacts_dir),
    }


def filter_tournament_matches(
    features_df: pd.DataFrame,
    tournament: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Filter features DataFrame to specific tournament window."""
    mask = (
        (features_df["tournament"] == tournament)
        & (features_df["date"] >= pd.Timestamp(start_date))
        & (features_df["date"] <= pd.Timestamp(end_date))
    )
    return features_df.loc[mask].copy()


def predict_single_match(
    match_row: pd.DataFrame,
    artifacts: dict,
) -> dict:
    """Produce predictions for a single match.

    Returns dict with keys:
    - xgb_probs: dict with home_win_prob, draw_prob, away_win_prob
    - xgb_pick: str (home_win/draw/away_win)
    - poisson_home_lambda: float
    - poisson_away_lambda: float
    - poisson_home_goals: int
    - poisson_away_goals: int
    """
    final_feature_columns = artifacts["final_feature_columns"]
    numeric_cols = [col for col in final_feature_columns if not col.startswith("tournament_tier_")]

    X = _build_feature_matrix(
        match_row, artifacts["encoder"], numeric_cols, final_feature_columns,
    )

    # XGBoost classifier prediction
    probs_raw = artifacts["clf"].predict_proba(X)
    xgb_probs = {
        "home_win_prob": float(probs_raw[0][2]),
        "draw_prob": float(probs_raw[0][1]),
        "away_win_prob": float(probs_raw[0][0]),
    }
    xgb_pick = max(xgb_probs, key=xgb_probs.get).replace("_prob", "")

    # Poisson score prediction
    X_scaled = artifacts["scaler"].transform(X)
    home_lambda = float(artifacts["home_model"].predict(X_scaled)[0])
    away_lambda = float(artifacts["away_model"].predict(X_scaled)[0])

    return {
        "xgb_probs": xgb_probs,
        "xgb_pick": xgb_pick,
        "poisson_home_lambda": home_lambda,
        "poisson_away_lambda": away_lambda,
        "poisson_home_goals": int(round(home_lambda)),
        "poisson_away_goals": int(round(away_lambda)),
    }


def generate_report(
    features_df: pd.DataFrame,
    artifacts: dict,
) -> str:
    """Generate the full validation report for WC 2022 and Euro 2024.

    Returns the report as a string.
    """
    lines: list[str] = []

    def _log(msg: str = "") -> None:
        lines.append(msg)

    _log("# Tournament Validation Report: World Cup 2022 & Euro 2024")
    _log()
    _log("Replays real tournament fixtures through the trained pipeline and compares")
    _log("predictions against actual results.")
    _log()

    all_matches: list[dict] = []

    for tconf in TOURNAMENT_CONFIGS:
        matches = filter_tournament_matches(
            features_df, tconf["tournament"], tconf["start"], tconf["end"],
        )
        _log(f"## {tconf['name']} ({len(matches)} matches)")
        _log()
        _log(f"| {'Date':<12} | {'Home':<20} | {'Away':<20} | {'Score':>5} | {'XGB Pick':>8} | {'Correct':>7} | {'Poisson':>7} | {'Exact':>5} |")
        _log(f"|{'-'*14}|{'-'*22}|{'-'*22}|{'-'*7}|{'-'*10}|{'-'*9}|{'-'*9}|{'-'*7}|")

        for idx, (_, match) in enumerate(matches.iterrows()):
            match_row = match.to_frame().T
            for col in match_row.columns:
                if col in ["date", "home_team", "away_team", "tournament", "outcome", "tournament_tier"]:
                    continue
                match_row[col] = pd.to_numeric(match_row[col], errors="coerce")

            preds = predict_single_match(match_row, artifacts)

            actual_home = int(match["home_score"])
            actual_away = int(match["away_score"])
            actual_outcome = match["outcome"]
            xgb_correct = preds["xgb_pick"] == actual_outcome
            poisson_exact = (
                preds["poisson_home_goals"] == actual_home
                and preds["poisson_away_goals"] == actual_away
            )

            date_str = pd.Timestamp(match["date"]).strftime("%Y-%m-%d")
            score_str = f"{actual_home}-{actual_away}"
            correct_str = "Yes" if xgb_correct else "No"
            exact_str = "Yes" if poisson_exact else "No"

            home_name = str(match["home_team"])[:20]
            away_name = str(match["away_team"])[:20]

            _log(
                f"| {date_str:<12} | {home_name:<20} | {away_name:<20} | {score_str:>5} | "
                f"{preds['xgb_pick']:>8} | {correct_str:>7} | "
                f"{preds['poisson_home_goals']}-{preds['poisson_away_goals']:>5} | {exact_str:>5} |"
            )

            all_matches.append({
                "tournament": tconf["name"],
                "date": match["date"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "home_score": actual_home,
                "away_score": actual_away,
                "actual_outcome": actual_outcome,
                "xgb_pick": preds["xgb_pick"],
                "xgb_correct": xgb_correct,
                "xgb_probs": preds["xgb_probs"],
                "poisson_home_goals": preds["poisson_home_goals"],
                "poisson_away_goals": preds["poisson_away_goals"],
                "poisson_exact": poisson_exact,
                "poisson_home_lambda": preds["poisson_home_lambda"],
                "poisson_away_lambda": preds["poisson_away_lambda"],
            })

        _log()

    # Summary statistics
    n = len(all_matches)
    y_true = pd.Series([m["actual_outcome"] for m in all_matches])
    y_pred_labels = pd.Series([m["xgb_pick"] for m in all_matches])
    y_pred_probs = pd.DataFrame([m["xgb_probs"] for m in all_matches])

    overall_accuracy = accuracy(y_true, y_pred_labels)
    overall_logloss = log_loss_score(y_true, y_pred_probs)
    overall_brier = brier_score(y_true, y_pred_probs)
    n_exact_score = sum(1 for m in all_matches if m["poisson_exact"])

    _log("## Summary Statistics")
    _log()
    _log(f"- **Matches evaluated**: {n}")
    _log(f"- **Classifier accuracy**: {overall_accuracy:.1%}")
    _log(f"- **Classifier log-loss**: {overall_logloss:.4f}")
    _log(f"- **Classifier Brier score**: {overall_brier:.4f}")
    _log(f"- **Exact scoreline matches (Poisson)**: {n_exact_score}/{n} ({n_exact_score/n:.1%})")
    _log()

    # Most confident correct predictions
    correct = [m for m in all_matches if m["xgb_correct"]]
    correct_sorted = sorted(correct, key=lambda m: max(m["xgb_probs"].values()), reverse=True)

    _log("### Top 3 Most Confident Correct Predictions")
    _log()
    _log(f"| {'Match':<45} | {'Actual':>5} | {'XGB Prob':>8} | {'Pick':>8} |")
    _log(f"|{'-'*47}|{'-'*7}|{'-'*10}|{'-'*10}|")
    for m in correct_sorted[:3]:
        match_str = f"{m['home_team']} vs {m['away_team']}"
        prob = max(m["xgb_probs"].values())
        _log(
            f"| {match_str:<45} | {m['home_score']}-{m['away_score']:>3} | "
            f"{prob:>8.1%} | {m['xgb_pick']:>8} |"
        )
    _log()

    # Most confident wrong predictions (upsets)
    wrong = [m for m in all_matches if not m["xgb_correct"]]
    wrong_sorted = sorted(wrong, key=lambda m: max(m["xgb_probs"].values()), reverse=True)

    _log("### Top 3 Most Confident Wrong Predictions (Biggest Upsets)")
    _log()
    _log(f"| {'Match':<45} | {'Actual':>5} | {'XGB Prob':>8} | {'Pick':>8} |")
    _log(f"|{'-'*47}|{'-'*7}|{'-'*10}|{'-'*10}|")
    for m in wrong_sorted[:3]:
        match_str = f"{m['home_team']} vs {m['away_team']}"
        prob = max(m["xgb_probs"].values())
        _log(
            f"| {match_str:<45} | {m['home_score']}-{m['away_score']:>3} | "
            f"{prob:>8.1%} | {m['xgb_pick']:>8} |"
        )
    _log()

    return "\n".join(lines)


def main() -> None:
    """Run the validation report and save to markdown."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = str(repo_root / "outputs" / "artifacts")
    models_dir = str(repo_root / "outputs" / "models")
    reports_dir = repo_root / "outputs" / "reports"

    features_df = pd.read_parquet(repo_root / "data" / "processed" / "features.parquet")
    logger.info("Loaded features: %d rows", len(features_df))

    artifacts = load_all_artifacts(artifacts_dir, models_dir)
    logger.info("Loaded all trained artifacts")

    report = generate_report(features_df, artifacts)

    # Print to console
    print(report)

    # Save as markdown
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "worldcup2022_euro2024_validation.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
