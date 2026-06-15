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
import os
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

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
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
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


def _to_jsonable(value: Any) -> Any:
    """Convert common Python / NumPy objects into JSON-serialisable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def extract_model_params(model: Any) -> Dict[str, Any]:
    """Return a JSON-friendly dictionary with the main model parameters."""
    if not hasattr(model, "get_params"):
        return {}

    params = model.get_params(deep=False)
    clean_params: Dict[str, Any] = {}
    for key, value in params.items():
        try:
            json.dumps(_to_jsonable(value))
            clean_params[key] = _to_jsonable(value)
        except TypeError:
            clean_params[key] = str(value)
    return clean_params


def append_experiment_log(
    record: Dict[str, Any],
    cfg: dict | None = None,
    filename: str = "experiment_log.csv",
) -> Path:
    """Append one experiment-traceability row to a local CSV fallback log."""
    cfg = cfg or CFG
    out_path = Path(cfg["paths"]["reports_metrics"]) / filename
    row = {key: _to_jsonable(value) for key, value in record.items()}

    if out_path.exists():
        existing = pd.read_csv(out_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    updated.to_csv(out_path, index=False)
    print(f"[train] Local experiment log updated: {out_path}")
    return out_path


def save_selected_model_metadata(
    metadata: Dict[str, Any],
    cfg: dict | None = None,
    filename: str = "selected_model_metadata.json",
) -> Path:
    """Persist selected-model metadata for reproducible inference/explainability."""
    cfg = cfg or CFG
    out_path = Path(cfg["paths"]["models_selected"]) / filename
    payload = _to_jsonable(metadata)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[train] Selected-model metadata saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 5. MLflow experiment tracking (optional)
# ---------------------------------------------------------------------------

def log_to_mlflow(
    model: Any,
    model_name: str,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    cfg: dict | None = None,
    scenario_name: str | None = None,
    test_metrics: Dict[str, float] | None = None,
    feature_columns: list[str] | None = None,
    model_path: str | Path | None = None,
    artifact_paths: list[str | Path] | None = None,
    extra_params: Dict[str, Any] | None = None,
    tags: Dict[str, Any] | None = None,
    run_name: str | None = None,
) -> Dict[str, Any]:
    """Log model, parameters, and metrics to an MLflow experiment.

    Returns a status dictionary so notebooks can communicate whether MLflow
    tracking succeeded or whether local CSV/JSON fallback should be used.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except Exception as exc:
        message = f"MLflow tracking unavailable; local CSV/JSON fallback should be used. Reason: {exc}"
        print(f"[train] {message}")
        return {"status": "fallback_local", "message": message}

    cfg = cfg or CFG
    mlflow_cfg = cfg.get("mlflow", {})
    project_name = cfg.get("project", {}).get("name", "minsur-quality-prediction")

    configured_experiment_name = mlflow_cfg.get("experiment_name")
    legacy_names = {"minsur", "minsur-silica-prediction"}
    if configured_experiment_name and configured_experiment_name not in legacy_names:
        experiment_name = configured_experiment_name
    else:
        experiment_name = "MINSUR Silica Prediction - Temporal Modeling"

    tracking_uri_cfg = str(mlflow_cfg.get("tracking_uri", "mlruns"))

    # MLflow 3.x blocks file-store by default. For local notebook/project workflows
    # we explicitly opt in when using a filesystem URI.
    if "://" not in tracking_uri_cfg:
        local_tracking_path = Path(cfg["paths"].get("mlruns", tracking_uri_cfg))
        local_tracking_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(str(local_tracking_path))
    else:
        mlflow.set_tracking_uri(tracking_uri_cfg)

    mlflow.set_experiment(experiment_name)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = run_name or f"{project_name} | {scenario_name or 'scenario-na'} | {model_name} | {run_timestamp}"
    artifact_paths = artifact_paths or []
    extra_params = extra_params or {}
    tags = tags or {}

    try:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("project", project_name)
            mlflow.set_tag("model_name", model_name)
            if scenario_name is not None:
                mlflow.set_tag("scenario", scenario_name)
            for key, value in tags.items():
                mlflow.set_tag(str(key), _to_jsonable(value))

            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_class", type(model).__name__)
            if scenario_name is not None:
                mlflow.log_param("scenario", scenario_name)
            if feature_columns is not None:
                mlflow.log_param("n_features", len(feature_columns))
                mlflow.log_dict({"feature_columns": feature_columns}, "feature_columns.json")
            if model_path is not None:
                mlflow.log_param("saved_model_path", str(model_path))

            for key, value in extract_model_params(model).items():
                mlflow.log_param(f"model_param__{key}", value)
            for key, value in extra_params.items():
                mlflow.log_param(str(key), _to_jsonable(value))

            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", float(v))
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", float(v))
            if test_metrics:
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", float(v))

            for artifact_path in artifact_paths:
                artifact_path = Path(artifact_path)
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))

            try:
                mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception:
                pass

        print(f"[train] MLflow run logged for: {model_name}")
        return {
            "status": "mlflow_logged",
            "experiment_name": experiment_name,
            "run_name": run_name,
            "run_id": run.info.run_id,
        }
    except Exception as exc:
        message = f"MLflow tracking failed; local CSV/JSON fallback should be used. Reason: {exc}"
        print(f"[train] {message}")
        return {"status": "fallback_local", "message": message}
