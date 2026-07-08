# Football Match Outcome & Score Prediction

This project builds a temporally-valid machine learning pipeline for predicting international football match outcomes and scorelines using historical match data. The implementation is designed to avoid leakage by building features strictly from information available before each match date.

## What is implemented

The repository now includes a modular Python package under [src/football_predictor](src/football_predictor) covering:

- Data loading for results, goalscorers, shootouts, and former names
- Data cleaning with null-score filtering, chronological ordering, duplicate handling, and conflict logging
- Team name resolution for historical identity mapping
- A chronological Elo rating engine with pre/post-match ratings and tournament-tier K-factor mapping
- Feature-building scaffolding for future rolling and match-context features
- Baseline, outcome, and score-model modules with evaluation support
- Reusable artifact I/O helpers and a CLI pipeline entrypoint

## Elo engine

The Elo implementation in [src/football_predictor/features/elo.py](src/football_predictor/features/elo.py) follows the requested football Elo specification:

- Home/away expected scores derived from rating differential and home advantage
- Goal-difference multiplier applied as specified
- Chronological forward-pass updates that prevent leakage
- Pre-match ratings stored as `home_elo_pre` and `away_elo_pre`
- Post-match ratings stored as `home_elo_post` and `away_elo_post`
- Tournament-tier K-factor lookup configured in [config/config.yaml](config/config.yaml)

## Project structure

- [config](config) — project configuration and YAML settings
- [data/raw](data/raw) — untouched raw CSV inputs
- [data/interim](data/interim) — intermediate cleaned data
- [data/processed](data/processed) — cleaned and feature-engineered training tables
- [src/football_predictor](src/football_predictor) — package modules for data loading, cleaning, feature engineering, modeling, evaluation, and utilities
- [tests](tests) — regression tests for core modules
- [scripts](scripts) — pipeline and validation entrypoints
- [outputs](outputs) — persisted artifacts, trained models, reports, and figures

## Setup

1. Create and activate a virtual environment.
2. Install the package and dependencies:

   ```bash
   .venv/bin/python -m pip install -e .
   ```

3. Run the test suite:

   ```bash
   .venv/bin/python -m pytest
   ```

4. Run the pipeline entrypoint:

   ```bash
   .venv/bin/python scripts/run_pipeline.py
   ```

## Notes

The project follows a temporal, leakage-safe design where features are built using only information available before each match date. This is especially important for Elo and other form-based features.
