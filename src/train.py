"""
src/train.py
============
Model training, registration, and persistence.

All models are trained using only X_train / y_train.
Validation metrics are computed on X_val / y_val (not test — test stays unseen
until final evaluation in evaluate.py / notebooks).
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

try:
    from .config import CFG
    from .evaluate import compute_metrics
except ImportError:
    from config import CFG
    from evaluate import compute_metrics

# Optional heavy models — imported lazily so the project works without them
try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Model factory
# ---------------------------------------------------------------------------

def get_models(cfg: dict | None = None) -> Dict[str, Any]:
    """Return a dictionary of untrained model instances.

    All models use a fixed random_state for reproducibility.
    Hyperparameters come from config.yaml — no hardcoded values here.
    """
    cfg = cfg or CFG
    mcfg = cfg["models"]
    rs = mcfg["random_state"]

    models: Dict[str, Any] = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=mcfg["ridge"]["alpha"]),
        "Random Forest": RandomForestRegressor(
            n_estimators=mcfg["random_forest"]["n_estimators"],
            max_depth=mcfg["random_forest"]["max_depth"],
            min_samples_leaf=mcfg["random_forest"]["min_samples_leaf"],
            n_jobs=mcfg["random_forest"]["n_jobs"],
            random_state=rs,
        ),
    }

    if _XGBOOST_AVAILABLE:
        xgb_cfg = mcfg["xgboost"]
        models["XGBoost"] = XGBRegressor(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            max_depth=xgb_cfg["max_depth"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            n_jobs=xgb_cfg["n_jobs"],
            random_state=rs,
            verbosity=0,
        )
    else:
        print("[train] XGBoost not available — skipping.")

    if _LIGHTGBM_AVAILABLE:
        lgb_cfg = mcfg["lightgbm"]
        models["LightGBM"] = LGBMRegressor(
            n_estimators=lgb_cfg["n_estimators"],
            learning_rate=lgb_cfg["learning_rate"],
            num_leaves=lgb_cfg["num_leaves"],
            n_jobs=lgb_cfg["n_jobs"],
            random_state=rs,
            verbosity=-1,
        )
    else:
        print("[train] LightGBM not available — skipping.")

    return models


# ---------------------------------------------------------------------------
# 2. Train a single model
# ---------------------------------------------------------------------------

def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    model_name: str = "",
) -> Tuple[Any, Dict[str, float], Dict[str, float]]:
    """Fit a model and return train / val metrics.

    For XGBoost with early_stopping_rounds, we pass eval_set from validation.

    Returns
    -------
    model : fitted estimator
    train_metrics : dict with MAE, RMSE, R2 on train
    val_metrics   : dict with MAE, RMSE, R2 on val (empty if val not provided)
    """
    # XGBoost: pass eval_set for early stopping
    if _XGBOOST_AVAILABLE and isinstance(model, XGBRegressor) and X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    train_metrics = compute_metrics(y_train, train_preds)

    val_metrics: Dict[str, float] = {}
    if X_val is not None and y_val is not None:
        val_preds = model.predict(X_val)
        val_metrics = compute_metrics(y_val, val_preds)

    if model_name:
        print(f"[train] {model_name} | train: {train_metrics} | val: {val_metrics}")

    return model, train_metrics, val_metrics


# ---------------------------------------------------------------------------
# 3. Train all models
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: dict | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
    """Train all models and return fitted estimators and metrics.

    Parameters
    ----------
    X_train, y_train : training split
    X_val, y_val     : validation split (used for evaluation, NOT for fitting)

    Returns
    -------
    fitted_models : {model_name: fitted_estimator}
    all_metrics   : {model_name: {"train": metrics_dict, "val": metrics_dict}}
    """
    cfg = cfg or CFG
    models = get_models(cfg)
    fitted_models: Dict[str, Any] = {}
    all_metrics: Dict[str, Dict] = {}

    for name, model in models.items():
        print(f"\n[train] Fitting: {name}")
        fitted, tr_metrics, val_metrics = train_model(
            model, X_train, y_train, X_val, y_val, model_name=name
        )
        fitted_models[name] = fitted
        all_metrics[name] = {"train": tr_metrics, "val": val_metrics}

    return fitted_models, all_metrics


# ---------------------------------------------------------------------------
# 4. Model persistence
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    model_name: str,
    folder: str = "selected",
    cfg: dict | None = None,
    filename: str = "model.pkl",
) -> Path:
    """Persist a fitted model using joblib.

    Parameters
    ----------
    model : fitted scikit-learn compatible estimator
    model_name : human-readable name (saved in metadata file)
    folder : "selected" or "baseline"
    cfg : project config
    filename : output filename (default model.pkl)

    Returns
    -------
    Path to the saved model file.
    """
    cfg = cfg or CFG
    folder_key = f"models_{folder}"
    out_dir = Path(cfg["paths"][folder_key])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / filename
    meta_path = out_dir / "model_meta.txt"

    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"class: {type(model).__name__}\n")

    print(f"[train] Model saved to: {model_path}")
    return model_path


def load_model(folder: str = "selected", cfg: dict | None = None, filename: str = "model.pkl") -> Any:
    """Load a persisted model."""
    cfg = cfg or CFG
    folder_key = f"models_{folder}"
    model_path = Path(cfg["paths"][folder_key]) / filename
    model = joblib.load(model_path)
    print(f"[train] Model loaded from: {model_path}")
    return model


# ---------------------------------------------------------------------------
# 5. MLflow experiment tracking (optional)
# ---------------------------------------------------------------------------

def log_to_mlflow(
    model: Any,
    model_name: str,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    cfg: dict | None = None,
) -> None:
    """Log model, parameters, and metrics to an MLflow experiment.

    Fails silently if mlflow is not installed.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("[train] mlflow not installed — skipping tracking.")
        return

    cfg = cfg or CFG
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(str(Path(cfg["paths"]["mlruns"]).parent / mlflow_cfg.get("tracking_uri", "mlruns")))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "minsur"))

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_class", type(model).__name__)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception:
            pass  # some models (e.g. LightGBM) need mlflow.lightgbm
    print(f"[train] MLflow run logged for: {model_name}")
