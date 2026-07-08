# Football Match Outcome & Score Prediction

This project builds a temporally-valid machine learning pipeline for predicting international football match outcomes (home win / draw / away win) and scorelines using historical match data.

## Project goals

- Load and clean raw international football datasets from the local CSV files.
- Build chronological features such as Elo ratings, rolling form, and rest-day signals.
- Train baseline and supervised models for outcome and score prediction.
- Persist reusable artifacts such as Elo snapshots, feature columns, and scalers for later inference.

## Repository structure

- config/ — project configuration and YAML settings
- data/raw/ — untouched raw CSV inputs
- data/interim/ — intermediate cleaned data
- data/processed/ — cleaned and feature-engineered training tables
- src/football_predictor/ — package modules for data loading, cleaning, feature engineering, modeling, evaluation, and utilities
- tests/ — regression tests for core modules
- outputs/artifacts/ — persisted artifacts for future prediction runs
- outputs/models/ — trained model artifacts
- outputs/reports/ — evaluation reports
- outputs/figures/ — plots and charts

## Current implementation status

Implemented modules include:

- data loading: results, goalscorers, shootouts, and former names with dtype enforcement and logging
- data cleaning: null-score removal, chronological sorting, duplicate resolution, and conflict exclusion
- name resolution: former-name lookup and team-name resolution with safe fallback logging
- cleaned results persistence: parquet export of deduplicated match data (49,492 rows from 49,502 raw)
- reusable artifact I/O helpers for saving/loading JSON or pickle artifacts
- comprehensive test coverage for all data-layer modules

## Data pipeline

The pipeline loads raw datasets, cleans them by removing null-score rows and resolving duplicates,
and persists the final table as `data/processed/results_cleaned.parquet`. Duplicates with matching
scores are resolved by keeping one row; conflicting duplicates are excluded entirely and logged as warnings.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies with:

   ```bash
   .venv/bin/python -m pip install -e .
   ```

3. Run tests with:

   ```bash
   .venv/bin/python -m pytest
   ```

## Notes

The project follows a temporal, leakage-safe design where features must be built using only information available before the match date.
