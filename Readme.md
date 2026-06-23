# FIFA World Cup 2026 — Match Prediction Pipeline

An end-to-end data pipeline that collects live FIFA World Cup data, engineers
features from current and historical tournaments, trains a match-outcome
prediction model, runs a Monte Carlo tournament simulation, and serves
everything through an interactive Streamlit dashboard.

---

## Pipeline overview

| Phase | Script | What it does |
|---|---|---|
| 1 | `datacollection.py` | Pulls 2026 matches (football-data.org) and rosters (Zafronix API), plus historical World Cup data (2014/2018/2022) for training |
| 2 | `featureengineering.py` | Builds per-team and per-match features across all years, handling schema differences between tournaments |
| 3 | `eda.py` | Generates exploratory plots (squad strength, goal distributions, feature correlations, year-over-year comparisons) |
| 4 | `modeltraining.py` | Trains Logistic Regression / Random Forest / XGBoost, picks the best by cross-validated accuracy, fits a Poisson model for score prediction |
| 5 | `simulator.py` | Runs 1,000+ Monte Carlo simulations of the remaining tournament bracket for championship and qualification odds |
| 6 | `dashboard.py` | Streamlit app: match predictions, championship odds, head-to-head team comparison |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
FOOTBALL_DATA_API_KEY=your_key_here
ZAFRONIX_API_KEY=your_key_here
```

- Free key for football-data.org: https://www.football-data.org/client/register
- Free key for Zafronix: https://api.zafronix.com/signup

## Running the pipeline

Run phases in order — each one depends on outputs from the previous step:

```bash
python src/datacollection.py
python src/featureengineering.py
python src/eda.py
python src/modeltraining.py
python src/simulator.py
streamlit run src/dashboard.py
```

Re-run from `datacollection.py` onward periodically during the tournament to
pick up new results — `modeltraining.py` automatically drops any
"upcoming match" rows that have already been played if the data is stale.

## Data sources

- **2026 matches:** [football-data.org](https://www.football-data.org) (free tier)
- **Player rosters (2026 + historical):** [Zafronix World Cup API](https://api.zafronix.com) (free tier, 250 req/day)
- **FIFA rankings:** hardcoded snapshot (June 2026) — no reliable free historical ranking source was found for past tournaments, so ranking features only apply to current-year matches (see limitations)

## Possible next steps

- Source real point-in-time historical FIFA rankings (available on
  Wikipedia per-tournament pages) to unlock the model's strongest feature
  for historical training rows
- Add more historical tournaments to grow the training set further
- Replace the random bracket pairing with FIFA's actual seeding rules
- Source a real squad-strength rating (e.g. from a ratings provider) as a
  more direct quality signal than age/composition proxies

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, Plotly, matplotlib

## Disclaimer

Built as a personal/learning project. Not affiliated with FIFA, the World
Cup, or any data provider referenced above. All predictions are
probabilistic estimates from a small dataset and should be treated as
exploratory, not authoritative.
