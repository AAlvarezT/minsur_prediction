"""
src/predict.py
==============
Inference and what-if simulation utilities.

These functions are used in notebook 04 and for any production inference pipeline.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import CFG
from src.train import load_model


# ---------------------------------------------------------------------------
# 1. Single-observation inference
# ---------------------------------------------------------------------------

def predict_single(model, X_row: pd.DataFrame) -> float:
    """Return the model's prediction for a single observation.

    Parameters
    ----------
    model : fitted scikit-learn compatible estimator
    X_row : pd.DataFrame with exactly 1 row and the correct feature columns

    Returns
    -------
    float : predicted % Silica Concentrate
    """
    pred = model.predict(X_row)
    return float(pred[0])


# ---------------------------------------------------------------------------
# 2. What-if scenario builder
# ---------------------------------------------------------------------------

def build_scenario(
    base_row: pd.DataFrame,
    modifications: Dict[str, float],
) -> pd.DataFrame:
    """Create a modified copy of a reference observation.

    Parameters
    ----------
    base_row : pd.DataFrame, 1-row reference observation
    modifications : dict mapping column_name -> new_value
        Values can be absolute (e.g. {"Ore Pulp pH": 10.2}) or
        computed externally as percentages before calling this function.

    Returns
    -------
    pd.DataFrame : 1-row modified scenario
    """
    scenario = base_row.copy()
    for col, val in modifications.items():
        if col in scenario.columns:
            scenario[col] = val
        else:
            print(f"[predict] Warning: column '{col}' not in feature set — skipped.")
    return scenario


def apply_pct_change(
    base_row: pd.DataFrame,
    column: str,
    pct_change: float,
) -> pd.DataFrame:
    """Apply a percentage change to a single column of a reference row.

    Parameters
    ----------
    base_row : pd.DataFrame, 1-row reference
    column : column to modify
    pct_change : e.g. +5.0 means +5%, -5.0 means -5%

    Returns
    -------
    pd.DataFrame : modified row
    """
    scenario = base_row.copy()
    if column in scenario.columns:
        original = float(scenario[column].iloc[0])
        scenario[column] = original * (1 + pct_change / 100)
    else:
        print(f"[predict] Warning: column '{column}' not found.")
    return scenario


# ---------------------------------------------------------------------------
# 3. Scenario comparison
# ---------------------------------------------------------------------------

def compare_scenarios(
    model,
    base_row: pd.DataFrame,
    scenarios: List[Dict],
) -> pd.DataFrame:
    """Predict % Silica Concentrate for the base and all alternative scenarios.

    Parameters
    ----------
    model : fitted estimator
    base_row : pd.DataFrame, 1-row reference
    scenarios : list of dicts, each with keys:
        - "name": str, scenario label
        - "description": str, human-readable change description
        - "row": pd.DataFrame, the modified observation

    Returns
    -------
    pd.DataFrame with columns:
        Scenario, Description, Predicted_Silica, Delta_vs_Base, Delta_pct_vs_Base
    """
    base_pred = predict_single(model, base_row)
    rows = [
        {
            "Scenario": "Base",
            "Description": "No changes (reference observation)",
            "Predicted_Silica": round(base_pred, 4),
            "Delta_vs_Base": 0.0,
            "Delta_pct_vs_Base": 0.0,
        }
    ]

    for sc in scenarios:
        pred = predict_single(model, sc["row"])
        delta = pred - base_pred
        delta_pct = (delta / base_pred * 100) if base_pred != 0 else np.nan
        rows.append(
            {
                "Scenario": sc["name"],
                "Description": sc["description"],
                "Predicted_Silica": round(pred, 4),
                "Delta_vs_Base": round(delta, 4),
                "Delta_pct_vs_Base": round(delta_pct, 2),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Sensitivity sweep (1-D)
# ---------------------------------------------------------------------------

def sensitivity_sweep(
    model,
    base_row: pd.DataFrame,
    column: str,
    pct_range: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute predictions as one variable sweeps across a range of pct changes.

    Useful for plotting the sensitivity of % Silica Concentrate to a single
    operational variable.

    Parameters
    ----------
    model : fitted estimator
    base_row : 1-row reference
    column : feature to vary
    pct_range : array of percentage changes to apply, e.g. np.linspace(-20, 20, 41)

    Returns
    -------
    pd.DataFrame with columns: pct_change, feature_value, predicted_silica
    """
    if pct_range is None:
        pct_range = np.linspace(-20, 20, 41)

    results = []
    base_val = float(base_row[column].iloc[0])

    for pct in pct_range:
        modified = apply_pct_change(base_row, column, pct)
        pred = predict_single(model, modified)
        results.append({
            "pct_change": round(pct, 1),
            "feature_value": round(base_val * (1 + pct / 100), 4),
            "predicted_silica": round(pred, 4),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. Save scenario results
# ---------------------------------------------------------------------------

def save_scenario_table(table: pd.DataFrame, cfg: dict | None = None, filename: str = "scenario_comparison.csv") -> Path:
    """Save the scenario comparison table to reports/metrics/."""
    cfg = cfg or CFG
    out_path = Path(cfg["paths"]["reports_metrics"]) / filename
    table.to_csv(out_path, index=False)
    print(f"[predict] Scenario table saved to: {out_path}")
    return out_path
