# Mining Process Quality Prediction

**Minsur Analytics Technical Challenge**  
*Senior Analytics / Data Science Role*

---

## Business Objective

In the Minsur flotation plant, the quality of the final iron ore concentrate is determined by its **% Silica Concentrate** — a lower value indicates purer iron concentrate and greater commercial value. Laboratory quality measurements are available only with a significant delay (up to several hours after the corresponding sensor readings), which prevents real-time operational responses to quality deviations.

**Goal:** Build a machine learning model that predicts `% Silica Concentrate` in real time using continuous process sensor data (feed quality, reagent flows, flotation column parameters), enabling operators to detect and respond to quality excursions before laboratory results become available.

---

## Dataset

**Source:** [Quality Prediction in a Mining Process — Kaggle](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)

- ~737,000 observations
- Sampling interval: ~20 minutes (sensor data), ~1 hour (lab measurements)
- Time range: March 2017 – September 2017
- 24 columns: date, feed characteristics, reagent flows, flotation column parameters, lab measurements

---

## Target Variable

| Variable | Description |
|---|---|
| `% Silica Concentrate` | Percentage of silica in the final iron ore concentrate. Lower = better quality. |

---

## Approach

### Temporal Integrity (Critical Design Decision)

This is a **time-series prediction problem**. All data splits are strictly temporal — no random shuffling is ever applied. Reasons:

1. A random split would allow training on future observations to predict past ones — a form of data leakage that produces over-optimistic evaluation metrics.
2. The deployed model operates in production by predicting the present/future, never the past.
3. Temporal ordering also ensures that lag and rolling features only use information available at inference time.

### Leakage Prevention

- `% Iron Concentrate` and `% Silica Concentrate` are lab measurements with delay → **excluded from features**.
- Only **lagged versions** of these variables (shift ≥ 1) are permitted.
- Rolling statistics are computed on `shift(1)` of each series to exclude the current row.
- Scalers are fitted exclusively on the training split.

---

## Feature Engineering

| Feature type | Description |
|---|---|
| **Raw operational features** | % Iron Feed, % Silica Feed, Starch Flow, Amina Flow, Ore Pulp Flow, pH, Density, 14 flotation column variables |
| **Temporal features** | hour, day, month, dayofweek, is_weekend |
| **Lag features** | 1, 3, 6, 12 rows back (~20 min to ~4 h) |
| **Rolling mean** | windows 3, 6, 12 rows |
| **Rolling std** | windows 3, 6, 12 rows |
| **Diff features** | period-over-period differences |

---

## Models Evaluated

| Model | Description |
|---|---|
| **Baseline (Mean)** | Predict training mean for all observations |
| **Ridge** | Regularised linear regression (α=1.0) |
| **Random Forest** | Ensemble of 300 trees, max_depth=10 |
| **XGBoost** | Gradient boosting with early stopping |
| **LightGBM** | Fast gradient boosting |

All models are evaluated on the **validation set** (temporal). Final evaluation is performed once on the **test set** after model selection is complete.

---

## Metrics

| Metric | Formula | Reason |
|---|---|---|
| **MAE** | mean \|y − ŷ\| | Directly interpretable in % silica units |
| **RMSE** | √mean(y − ŷ)² | Penalises large quality excursions more heavily |
| **R²** | 1 − SS_res/SS_tot | Proportion of variance explained |

---

## Model Selection

The final model is selected based on **lowest RMSE on the validation set**.  
Justification: In a flotation process, large prediction errors are disproportionately costly (they may lead to off-spec product or unnecessary reagent waste), so penalising large errors more heavily via RMSE is operationally appropriate.

The selected model is saved to `models/selected/model.pkl`.

---

## Explainability (Level 3)

SHAP (SHapley Additive exPlanations) is used to explain the selected model's behaviour:

- **Global**: Summary beeswarm and bar plots showing top 20 features by mean |SHAP|.
- **Dependence plots**: How individual operational variables relate to predicted silica.
- **Local (waterfall)**: Per-observation explanation for high/low silica predictions.

### Key operational drivers (model-learned associations, not causal claims)

| Variable | Effect direction | Operational interpretation |
|---|---|---|
| `% Silica Feed` | ↑ → higher silica | Feed quality directly propagates to concentrate |
| `% Iron Feed` | ↑ → lower silica | Richer ore → cleaner separation |
| `Amina Flow` | ↑ → lower silica | More collector → better silica suppression |
| `Starch Flow` | context-dependent | Depressant: excess can reduce selectivity |
| `Ore Pulp pH` | optimal range → lower | pH controls reagent efficiency |
| Flotation column air flow | excess → higher silica | Mechanical silica entrainment |

> **Disclaimer:** SHAP values describe the model's behaviour, not the underlying process physics. Operational decisions require domain expertise validation.

---

## Prediction Simulations (Level 4)

Notebook 04 demonstrates what-if scenario analysis using a reference observation from the test set. Scenarios include:

| Scenario | Modification |
|---|---|
| A | Amina Flow +5% |
| B | Amina Flow −5% |
| C | Starch Flow −5% |
| D | Starch Flow +5% |
| E | Ore Pulp pH +0.5 |
| F | Ore Pulp pH −0.5 |
| G | % Silica Feed −10% |
| H | Combined: Amina +5% + Starch −5% |

Sensitivity sweeps visualise the model's response to continuous variation in Amina Flow and pH.

> **Disclaimer:** These are predictive simulations, not engineering recommendations. Results are valid only within the historical data distribution.

---

## Limitations

1. **Temporal scope**: Model trained on ~7 months of data. Seasonal or longer-term process changes may require retraining.
2. **No mechanistic modelling**: The model learns statistical associations, not physical laws. It cannot extrapolate reliably outside the training distribution.
3. **Point predictions only**: No uncertainty quantification. A production system should provide prediction intervals.
4. **Lag consistency in inference**: At deployment, lag features must be populated from real historical process data — a proper feature store is needed.
5. **Recovery trade-off**: The model predicts silica but does not model iron recovery. Optimising silica alone may worsen overall metallurgical performance.
6. **Lab measurement delay handling**: This model does not explicitly model the delay in lab readings — it simply excludes them. A more sophisticated approach could model the delay distribution explicitly.

---

## Project Structure

```
minsur-quality-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── data/
│   ├── raw/                    ← Place Kaggle CSV here
│   ├── interim/                ← Cleaned data (data_cleaned.parquet)
│   └── processed/              ← Train/val/test splits (parquet)
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_feature_engineering_modeling.ipynb
│   ├── 03_explainability.ipynb
│   └── 04_simulation_what_if.ipynb
├── src/
│   ├── config.py               ← YAML loader + path resolver
│   ├── data_preprocessing.py   ← Load, clean, validate raw data
│   ├── feature_engineering.py  ← Lags, rolling stats, temporal split
│   ├── train.py                ← Model factory, training, persistence
│   ├── evaluate.py             ← Metrics, residual analysis
│   ├── explain.py              ← SHAP utilities
│   └── predict.py              ← Inference + what-if scenarios
├── models/
│   ├── baseline/
│   └── selected/               ← model.pkl + feature_columns.json
├── reports/
│   ├── figures/                ← All generated plots
│   └── metrics/                ← model_comparison.csv, scenario_comparison.csv
└── mlruns/                     ← MLflow experiment tracking
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the Kaggle dataset in data/raw/
# (or let notebook 01 download it via kagglehub)

# 3. Run notebooks in order
# Open notebooks/ in VS Code and run cells top-to-bottom:
#   01_data_understanding.ipynb
#   02_feature_engineering_modeling.ipynb
#   03_explainability.ipynb
#   04_simulation_what_if.ipynb
```

---

## MLflow Tracking

Model runs are logged to `mlruns/`. To view the MLflow UI:

```bash
cd minsur-quality-prediction
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

Note: in `mlruns/`, folder names are internal MLflow IDs by design. Use the MLflow UI to see descriptive experiment names and run names.

---

## Requirements

See `requirements.txt`. Key dependencies:

- `pandas`, `numpy`, `scikit-learn` — core ML
- `xgboost`, `lightgbm` — gradient boosting
- `shap` — explainability
- `mlflow` — experiment tracking
- `matplotlib`, `seaborn` — visualisation
- `joblib` — model persistence
- `kagglehub` — dataset download
