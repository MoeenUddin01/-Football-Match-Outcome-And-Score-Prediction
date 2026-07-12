# ChronoPitch — Football Match Outcome & Score Prediction

A full-stack ML system that predicts international football match outcomes (win/draw/loss) and scorelines (Poisson goal models), using 49,502 historical matches from 1872–2026. Built with a modular Python backend and a Next.js frontend.

Features are constructed strictly from information available before each match's kickoff date, preventing temporal leakage. Elo ratings, rolling form stats, and head-to-head records are computed via a single chronological forward pass.

## Live demo

- **Frontend**: Deployed on Vercel (Next.js 14, TypeScript, Tailwind CSS)
- **Backend**: Deployed on Fly.io / Render (FastAPI + uvicorn, Docker)

## Project structure

```
├── config/config.yaml          # All hyperparameters, date splits, Elo params
├── data/raw/                   # Untouched CSV inputs (results, goalscorers, shootouts, former_names)
├── src/football_predictor/     # Core Python package
│   ├── data/                   # Loader, cleaner, name resolver
│   ├── features/               # Elo engine, rolling stats, feature builder
│   ├── models/                 # Elo baseline, XGBoost classifier, Poisson regressor
│   ├── evaluation/             # Metrics (log-loss, Brier, accuracy), temporal splitter
│   └── utils/                  # Artifact I/O helpers
├── backend/app.py              # FastAPI REST API
├── frontend/                   # Next.js 14 app (TypeScript, Tailwind)
├── scripts/                    # CLI tools: predict_match, validate, compute Elo artifacts
├── tests/                      # 22 Python test files + 1 frontend test
├── notebooks/                  # Exploratory data analysis
└── Dockerfile                  # Python 3.12-slim, uv, uvicorn
```

## Architecture

```
raw CSVs → loader → cleaner → name_resolver
       → features/elo.py         (chronological forward pass)
       → features/rolling_stats.py (chronological forward pass)
       → features/feature_builder.py (assembles final table → parquet)
       → models/baseline_elo.py      (Elo-implied win probabilities)
       → models/outcome_classifier.py (XGBoost multi-class)
       → models/score_regressor.py    (Poisson goal models)
       → evaluation/metrics.py
       → evaluation/splitter.py   (temporal train/val/test splits)
```

### Frontend pages

| Page | Description |
|---|---|
| **Predictor** | Select home/away teams and tournament → displays XGBoost probability bar chart, Poisson scoreline, and model disagreement notice |
| **Rankings** | All teams ranked by current Elo rating with search |
| **Scorers** | Top international goalscorers leaderboard with team/tournament filters |
| **Validation** | World Cup 2022 + Euro 2024 backtest results with per-match predictions |

## Setup

### Backend (Python 3.12+)

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Compute Elo artifacts (required before starting the server)
uv run python scripts/compute_elo_artifacts.py

# Start the FastAPI server
uvicorn backend.app:app --host 0.0.0.0 --port 7860
```

### Frontend (Node.js 18+)

```bash
cd frontend
npm install
npm run dev    # http://localhost:3000
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/teams` | Sorted list of all teams |
| `GET` | `/api/rankings` | Elo-ranked teams with ratings |
| `GET` | `/api/tournaments` | Tournament names and tier mappings |
| `POST` | `/api/predict` | XGBoost + Poisson prediction for a match |
| `GET` | `/api/top-scorers` | Filterable top goalscorers |
| `GET` | `/api/top-scorers/team/{name}` | Top scorers for a single team |
| `GET` | `/api/validation-report` | WC 2022 + Euro 2024 backtest |

## Configuration

All parameters live in `config/config.yaml`:

- **Temporal splits**: train ≤ 2018-12-31, validation 2019–2022, test 2023+
- **Elo**: initial=1500, home advantage=100, K-factor by tournament tier (friendly=10 → World Cup=60)
- **Rolling windows**: form, goals, and H2H over last 5 and 10 matches
- **Model**: XGBoost (max_depth=5, lr=0.05, 300 estimators), Poisson alpha=1.0

## Scripts

| Script | Description |
|---|---|
| `scripts/predict_match.py` | CLI tool to predict a single match (`--home`, `--away`, `--tournament`, `--neutral`) |
| `scripts/validate_worldcup2022.py` | Replays WC 2022 + Euro 2024 fixtures, generates markdown report |
| `scripts/compute_elo_artifacts.py` | Computes and saves current Elo ratings + tournament tier map |

## Deployment

- **Docker**: `Dockerfile` builds a Python 3.12-slim image, runs uvicorn on port 7860
- **Fly.io**: `fly.toml` config for `chronopitch-backend` (shared-cpu-1x, 256MB)
- **Render**: `render.yaml` for Docker-based deployment (free tier)
- **Vercel**: `frontend/vercel.json` routes `/api/*` to a Python serverless function

## Known limitations

**Neutral-venue home/away labeling is arbitrary.** For matches at neutral venues (e.g. World Cup, Euros), the `home_team` and `away_team` columns reflect draw order for kit selection, not geography. Home teams at neutral venues carry no genuine advantage signal, contributing to lower accuracy on tournament matches (~47% vs ~61% on validation). This is a property of the source data, not a pipeline bug.
