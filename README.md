# NYC TLC Yellow Taxi Fare Prediction

Predicts the total fare of a NYC yellow-taxi trip at booking time (before the
trip happens), using only information a rider or dispatcher would actually
have in advance: pickup/dropoff location, trip distance, pickup time, and a
few trip/vendor metadata fields. Trained on 2024–2025 TLC trip records,
validated with forward-chaining time-series cross-validation, and evaluated
on a genuinely held-out 2026 test period.

## Why this project is more than "train a regressor"

- **Pricing rules encoded from the actual TLC rate card, not just learned
  from history** — flat JFK↔Manhattan fares, LaGuardia/Newark surcharges,
  the Congestion Relief Zone fee (effective Jan 2025), and the legal-holiday
  rush-hour exemption are all deterministic features derived from the
  published regulation. The model doesn't need years of post-change data to
  "learn" a new fee — it's correct from day one.
- **A trustworthy, forward-time evaluation.** 5-fold forward-chaining
  (walk-forward) cross-validation, a held-out validation window
  (Nov–Dec 2025), and a real 2026 test set that is only ever scored once, at
  the very end. Reported error numbers reflect genuine future performance,
  not resampled-history overfitting.
- **Built-in drift monitoring**, not just a one-off model
  (`src/analyze.py`, `src/mitigate.py`, `src/mitigate_walkforward.py`):
  PSI/KS feature-drift checks and automatic retrain-on-drift simulation, so
  there's a concrete, data-driven signal for *when* to retrain.
- **A receipt, not a black box.** [`src/fare_breakdown.py`](src/fare_breakdown.py)
  decomposes a predicted fare into its rate-card components (base/flat fare,
  MTA tax, congestion surcharge, airport fees, ...) plus a labeled
  Discount/Premium adjustment justified by the model's own SHAP
  contributions — the pieces always sum exactly to the real prediction.

## Project structure

```
src/
  config.py                    Central config: paths, cleaning thresholds,
                                feature lists, model hyperparameters
  cli.py                       Single production entry point (see below)
  data/                        Loading + cleaning
  features/                    Feature engineering (domain.py has every
                                TLC rate-card component; engineer.py wires
                                it into the FeatureEngineer used everywhere)
  models/                      Training, evaluation, SHAP analysis, registry
  drift/                       Drift detection (PSI/KS + Evidently)
  tracking/                    Weights & Biases integration + sweep configs
  train.py                     Forward-chaining CV + final model training
  evaluate.py                  Scores the real 2026 test set (run once)
  sweep.py                     Two-phase (random -> Bayesian) hyperparameter sweep
  baseline.py                  Raw 7-feature reference model, no engineering
  analyze.py                   Segmented error analysis, SHAP-over-time, drift
  mitigate.py / mitigate_walkforward.py
                                Drift detection + retraining simulation
  fare_breakdown.py            Customer-facing fare decomposition (CLI demo)

scripts/
  download_data.py             Pulls TLC parquet files from the public S3 bucket
  build_zone_centroids.py      Builds taxi_zone_centroids.csv from TLC zone shapes

01_EDA_Main.ipynb              Exploratory data analysis notebook
.github/workflows/train.yml    CI: downloads data, trains, evaluates on every push
```

## Setup

```
pip install -r requirements.txt
python scripts/download_data.py   # downloads TLC parquet files into training_set/ and test_set/
```

Weights & Biases logging is optional — pass `--no-wandb` to any command
that supports it, or leave `WANDB_API_KEY` unset.

## Usage

Everything runs through one CLI:

```
python -m src.cli <command> [args...]
python -m src.cli <command> --help   # each command's own arguments
```

| Command | What it does |
|---|---|
| `train` | Forward-chaining CV across all 4 models, then trains the final model on the full training set |
| `evaluate` | Scores the saved model on the real 2026 test set (run once — this is the actual holdout) |
| `sweep` | Two-phase W&B hyperparameter sweep (random search, then Bayesian search narrowed around the best result) |
| `baseline` | Raw 7-feature reference model, for comparison against the engineered feature set |
| `analyze` | Segmented error analysis, TreeSHAP-over-time, drift checks (train/val data only) |
| `mitigate` | Drift detection + mitigation strategy on a single drift month |
| `mitigate-walkforward` | Walk-forward simulation comparing a frozen model vs. one that auto-retrains on drift |
| `explain-fare` | Prints a rate-card fare breakdown (with justified adjustment) for a few example trips |

Example:

```
python -m src.cli train --sample 150000 --tag my-run
python -m src.cli evaluate --tag my-run
python -m src.cli explain-fare --tag my-run
```

Model artifacts are saved under `models/<tag>/`; per-run logs (CV results,
error breakdowns, feature importance, drift reports) are saved under
`logs/` with the tag in the filename.

## Data

- Training: 2024–2025 monthly TLC yellow-taxi parquet files (`training_set/`)
- Test: 2026 Jan + Feb, held out from all development (`test_set/`)
- Reference data: TLC taxi zone lookup/centroids, and macro-economic series
  (S&P 500, USD index) tried as candidate features — see `macro_data/`

All fixed reference data and random seeds are committed alongside the code,
and `.github/workflows/train.yml` re-runs the full pipeline on every push —
the results here are reproducible from a clean checkout.
