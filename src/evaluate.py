"""
src/evaluate.py
===============
Model evaluation utilities.

Kept separate from train.py so evaluation logic can be reused across
notebooks, CLI scripts, and MLflow logging without coupling to training code.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .config import CFG
except ImportError:
    from config import CFG


# ---------------------------------------------------------------------------
# 1. Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R² for a set of predictions.

    Why these three metrics?
    - MAE: directly interpretable in the same units as % Silica Concentrate.
    - RMSE: penalises large errors more heavily — important for avoiding costly
      quality excursions.
    - R²: indicates the proportion of variance explained; useful for model
      ranking and communication with stakeholders.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ---------------------------------------------------------------------------
# 2. Residual analysis helpers
# ---------------------------------------------------------------------------

def residuals(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute signed residuals (y_true - y_pred)."""
    return np.asarray(y_true) - np.asarray(y_pred)


def residuals_dataframe(
    y_true: pd.Series,
    y_pred: np.ndarray,
    split_name: str = "test",
) -> pd.DataFrame:
    """Build a DataFrame with true values, predictions, and residuals.

    Parameters
    ----------
    y_true : pd.Series (must have a DatetimeIndex for temporal plots)
    y_pred : np.ndarray
    split_name : str

    Returns
    -------
    pd.DataFrame with columns: y_true, y_pred, residual, abs_error, split
    """
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        },
        index=y_true.index if hasattr(y_true, "index") else None,
    )
    df["residual"] = df["y_true"] - df["y_pred"]
    df["abs_error"] = df["residual"].abs()
    df["split"] = split_name
    return df


# ---------------------------------------------------------------------------
# 3. Metrics comparison table
# ---------------------------------------------------------------------------

def build_metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Convert a dict of {model_name: metrics_dict} to a sorted DataFrame.

    Parameters
    ----------
    results : dict
        Example: {"Baseline": {"MAE": 1.2, "RMSE": 1.5, "R2": 0.0},
                  "XGBoost": {"MAE": 0.3, "RMSE": 0.4, "R2": 0.92}}

    Returns
    -------
    pd.DataFrame sorted by RMSE ascending.
    """
    table = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    table = table.sort_values("RMSE").reset_index(drop=True)
    return table


def save_metrics_table(table: pd.DataFrame, cfg: dict | None = None) -> Path:
    """Save the metrics comparison table to reports/metrics/model_comparison.csv."""
    cfg = cfg or CFG
    out_path = Path(cfg["paths"]["reports_metrics"]) / "model_comparison.csv"
    table.to_csv(out_path, index=False)
    print(f"[evaluate] Metrics saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 4. Temporal error analysis
# ---------------------------------------------------------------------------

def error_by_time_block(
    residuals_df: pd.DataFrame,
    freq: str = "ME",
) -> pd.DataFrame:
    """Compute mean absolute error grouped by a time frequency.

    Parameters
    ----------
    residuals_df : pd.DataFrame (output of residuals_dataframe, with DatetimeIndex)
    freq : str, pandas offset alias (default "ME" = month end)

    Returns
    -------
    pd.DataFrame with columns: period, MAE, RMSE, n_obs
    """
    if not isinstance(residuals_df.index, pd.DatetimeIndex):
        raise ValueError("residuals_df must have a DatetimeIndex.")

    grouped = residuals_df.resample(freq).agg(
        MAE=("abs_error", "mean"),
        RMSE=("residual", lambda x: np.sqrt((x**2).mean())),
        n_obs=("y_true", "count"),
    )
    return grouped
