# CLAUDE.md — Football Match Outcome & Score Prediction

## Project Identity

This project builds a temporally-valid machine learning system that predicts
international football match outcomes (win/draw/loss) and scorelines
(Poisson goal models), benchmarked against a from-scratch Elo rating engine,
using the `international_results` dataset (49,502 matches, 1872–2026).

The deliverable is not a notebook. It is a modular, tested, reproducible
`src/` package that could pass review from a senior ML engineer. Every
component must be independently testable and independently correct.

---

## Non-Negotiable Engineering Standards

1. **Root-cause over plausible explanation.** If a metric looks wrong (e.g.
   98% accuracy, or Elo ratings that never change), the agent traces the
   actual cause in the code before proposing a fix. "That's probably just
   because..." is not an acceptable stopping point.
2. **No silent data mutation.** Any row dropped, any null filled, any team
   name remapped must be logged with a count and a reason. Cleaning steps
   must be idempotent and reversible in principle (raw data is never
   overwritten).
3. **Every module ships with tests.** No feature engineering function,
   model wrapper, or evaluation metric is considered done until it has a
   pytest test covering at least: the happy path, one edge case, and one
   known-answer case (a hand-computable example).
4. **Temporal leakage is treated as a critical bug, not a style issue.**
   See "Anti-Leakage Rules" below. Any code that computes a feature using
   information not available strictly before the match's kickoff date is
   a bug, full stop — regardless of whether it "improves the backtest."
5. **Config over hardcoding.** Date splits, Elo K-factor, home advantage
   constant, rolling window sizes, model hyperparameters all live in
   `config/config.yaml`, never hardcoded inline.
6. **Reproducibility.** Every random process (train/test shuffling within
   a fold, model initialization) takes an explicit seed from config.

---

## Anti-Leakage Rules (read this twice)

This is the most common way football prediction projects quietly cheat
without anyone noticing:

- **Chronological processing only.** Elo ratings, rolling form, head-to-head
  records, and rest-day counts must be computed via a single forward pass
  through matches sorted by `date`. A team's rating/features going into
  match N must reflect only matches 1..N-1.
- **No global fit-then-split.** Do not fit scalers, encoders, or imputers
  on the full dataset and then split into train/test. Fit on train only,
  apply to test.
- **Train/test split is temporal, not random.** Train on matches through
  2018-12-31, validate on 2019-01-01 through 2022-12-31, hold out
  2023-01-01 onward (including World Cup 2022 result data if it falls after
  the validation cutoff — confirm exact boundary during implementation)
  as the final untouched test set. Never use `train_test_split` with
  `shuffle=True` on match data.
- **Result-derived features are the usual leak vector.** `home_score` and
  `away_score` of the *current* match must never appear, directly or
  indirectly, in that match's own feature row.

---

## Data Contract

Source files live in `data/raw/` and are never modified:

| File | Rows | Key columns | Known issues |
|---|---|---|---|
| `results.csv` | 49,502 | date, home_team, away_team, home_score, away_score, tournament, city, country, neutral | 7 rows have null scores (unplayed/future fixtures — must be filtered before training, not imputed) |
| `goalscorers.csv` | 47,868 | date, home_team, away_team, team, scorer, minute, own_goal, penalty | 44 null `scorer`, 254 null `minute` — acceptable, not blocking |
| `shootouts.csv` | 681 | date, home_team, away_team, winner, first_shooter | 422 null `first_shooter` (undocumented historically) |
| `former_names.csv` | 37 | current, former, start_date, end_date | Used to resolve historical team identity — see name_resolver module |

Team identity resolution: `results.csv` already uses each team's *current*
name for historical matches (per dataset documentation), so `former_names.csv`
is primarily needed for cross-referencing external validation data (e.g. if
future scripts pull in outside sources that use old names), and for any
human-readable historical reporting. The resolver module must still be built
and tested since downstream scripts will depend on it.

---

## Architecture Summary

```
raw CSVs → data/loader.py → data/cleaner.py → data/name_resolver.py
        → features/elo.py (chronological pass)
        → features/rolling_stats.py (chronological pass)
        → features/feature_builder.py (assembles final table)
        → models/{baseline_elo, outcome_classifier, score_regressor}.py
        → evaluation/backtester.py (walk-forward, time-based)
        → evaluation/metrics.py
        → scripts/run_pipeline.py (CLI orchestration)
```

## Repository Structure

```text
football-outcome-prediction/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── features.parquet
├── src/
│   └── football_predictor/
│       ├── __init__.py
│       ├── data/
│       │   ├── loader.py
│       │   ├── cleaner.py
│       │   └── name_resolver.py
│       ├── features/
│       │   ├── elo.py
│       │   ├── rolling_stats.py
│       │   └── feature_builder.py
│       ├── models/
│       │   ├── baseline_elo.py
│       │   ├── outcome_classifier.py
│       │   └── score_regressor.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── backtester.py
│       ├── pipeline/
│       │   └── train_pipeline.py
│       └── utils/
│           ├── logging_config.py
│           └── io.py
├── tests/
│   ├── test_data/
│   ├── test_features/
│   ├── test_models/
│   └── test_evaluation/
├── scripts/
│   ├── run_pipeline.py
│   └── validate_worldcup2022.py
└── outputs/
    ├── artifacts/
    │   ├── team_elo_latest.json
    │   ├── tournament_tier_map.json
    │   ├── feature_columns.json
    │   └── scaler.pkl
    ├── models/
    ├── reports/
    └── figures/
```

## Model Plan

1. **Baseline**: Elo-implied win probability (logistic function of rating
   difference + home advantage constant). This is the number every other
   model must beat to be worth keeping.
2. **Outcome classifier**: Gradient-boosted trees (XGBoost or
   LightGBM — agent's choice, document why) predicting P(home win),
   P(draw), P(away win). Features include Elo differential, rolling form,
   head-to-head, rest days, tournament type, neutral venue flag.
3. **Score regressor**: Two independent Poisson regressions (home goals,
   away goals) using attack/defense strength features derived from rolling
   scoring rates and Elo.
4. **Validation**: walk-forward backtest across the temporal splits above,
   reported via log-loss, Brier score, accuracy, and calibration curves —
   not accuracy alone, since draw prediction is a known hard class.
5. **Narrative validation**: a dedicated script re-plays World Cup 2022 and
   Euro 2024 fixtures through the trained model and reports predicted vs.
   actual outcomes as a human-readable summary.

---

## Definition of Done (per module)

A module is complete only when:
- [ ] Function/class has type hints and docstrings
- [ ] Corresponding test file exists and passes
- [ ] No hardcoded paths or magic numbers (pulled from config)
- [ ] Logging emits row counts in/out for any filtering step
- [ ] Agent has stated, in plain language, what could still go wrong

---

## Working Agreement

Claude (in the chat orchestrator role) issues one implementation prompt at
a time to the coding agent. The coding agent implements, runs tests, and
reports back before the next prompt is issued. No prompt should ask for
more than one logical module at a time — this keeps root-cause debugging
tractable if something breaks.