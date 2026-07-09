"""Compute Elo ratings and save reusable artifacts.

This script loads the raw results data, cleans it, computes Elo ratings,
and saves two artifacts:
1. team_elo_latest.json - Each team's most recent Elo rating
2. tournament_tier_map.json - Tournament-to-K-factor-tier mapping
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

from football_predictor.data.cleaner import clean_results
from football_predictor.data.loader import load_results
from football_predictor.features.elo import (
    _load_config,
    _resolve_tier_name,
    compute_elo_ratings,
)
from football_predictor.utils.io import load_artifact, save_artifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_latest_elo_ratings(elo_results_df) -> dict[str, float]:
    """Extract the most recent Elo rating for each team from the Elo results.

    Parameters
    ----------
    elo_results_df : pd.DataFrame
        DataFrame with columns: date, home_team, away_team, home_elo_post, away_elo_post

    Returns
    -------
    dict[str, float]
        Dictionary mapping team names to their most recent Elo ratings
    """
    ratings: dict[str, float] = {}

    # Process matches in chronological order
    for _, row in elo_results_df.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        # Update ratings - last occurrence wins
        ratings[home_team] = float(row["home_elo_post"])
        ratings[away_team] = float(row["away_elo_post"])

    return ratings


def build_tournament_tier_map(config: dict | None = None) -> dict[str, str]:
    """Build a mapping from tournament names to their tier names.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary. If None, loads from config.yaml.

    Returns
    -------
    dict[str, str]
        Dictionary mapping tournament names to tier names
    """
    # Load all known tournaments from the config
    config_data = _load_config(config)
    k_factor_tiers = config_data.get("elo_params", {}).get("k_factor_tiers", {})

    # Build the inverse mapping from tier name to K-factor
    tier_to_k = {tier: k for tier, k in k_factor_tiers.items()}

    # Create the tournament-to-tier mapping based on _resolve_tier_name logic
    # We'll hardcode the known tournaments from the lookup table in elo.py
    tournament_tier_map = {
        "Friendly": "friendly",
        "Minor Tournament": "minor_tournament",
        "FIFA World Cup": "world_cup",
        "UEFA Euro": "continental_championship",
        "Copa America": "continental_championship",
        "Copa América": "continental_championship",
        "African Cup of Nations": "continental_championship",
        "AFC Asian Cup": "continental_championship",
        "Gold Cup": "continental_championship",
    }

    # Add qualifiers
    qualifier_tournaments = [
        "FIFA World Cup qualification",
        "UEFA Euro qualification",
        "Copa America qualification",
        "African Cup of Nations qualification",
        "AFC Asian Cup qualification",
        "Gold Cup qualification",
    ]

    for tournament in qualifier_tournaments:
        tournament_tier_map[tournament] = "qualifier"

    return tournament_tier_map


def main():
    """Main function to compute and save Elo artifacts."""
    # Load config
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    # Get paths from config
    raw_dir = config.get("data_paths", {}).get("raw_dir", "data/raw")
    artifacts_dir = config.get("data_paths", {}).get("artifacts_dir", "outputs/artifacts")

    # Convert relative paths to absolute
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir_abs = repo_root / raw_dir if not Path(raw_dir).is_absolute() else Path(raw_dir)
    artifacts_dir_abs = repo_root / artifacts_dir if not Path(artifacts_dir).is_absolute() else Path(artifacts_dir)

    # Load raw results
    logger.info("Loading raw results from %s", raw_dir_abs)
    raw_results = load_results(raw_dir_abs)

    # Clean results
    logger.info("Cleaning results data...")
    cleaned_results = clean_results(raw_results)

    # Compute Elo ratings
    logger.info("Computing Elo ratings...")
    elo_results = compute_elo_ratings(cleaned_results, config=config)

    # Extract latest ratings
    logger.info("Extracting latest Elo ratings for each team...")
    latest_ratings = extract_latest_elo_ratings(elo_results)

    # Save team_elo_latest.json
    logger.info("Saving team_elo_latest.json to %s", artifacts_dir_abs)
    save_artifact(latest_ratings, "team_elo_latest.json", artifacts_dir_abs)

    # Build and save tournament tier map
    logger.info("Building tournament tier map...")
    tournament_tier_map = build_tournament_tier_map(config)

    logger.info("Saving tournament_tier_map.json to %s", artifacts_dir_abs)
    save_artifact(tournament_tier_map, "tournament_tier_map.json", artifacts_dir_abs)

    # Print summary
    logger.info("=== Summary ===")
    logger.info("Teams with latest Elo ratings: %d", len(latest_ratings))
    logger.info("Tournament tier mappings: %d", len(tournament_tier_map))

    # Show top 5 teams by rating
    top_teams = sorted(latest_ratings.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("Top 5 teams by Elo rating:")
    for team, rating in top_teams:
        logger.info("  %s: %.1f", team, rating)

    return 0


if __name__ == "__main__":
    sys.exit(main())
