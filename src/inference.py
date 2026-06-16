from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .app_utils import safe_load_json, safe_load_model
from .config import (
    TARGET_NAME,
    available_model_roles,
    get_model_role_config,
    get_project_root,
)

TARGET_LAG_PREFIX = "% Silica Concentrate_lag_"
TEMPORAL_FEATURES = {"hour", "day", "month", "dayofweek", "is_weekend"}


def load_model_metadata(model_role: str) -> dict[str, Any]:
    config = get_model_role_config(model_role)
    metadata_path = Path(config["metadata_path"])
    payload, err = safe_load_json(metadata_path)
    if err:
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=16)
def load_feature_columns(model_role: str) -> list[str]:
    config = get_model_role_config(model_role)
    feature_columns_path = Path(config["feature_columns_path"])
    payload, err = safe_load_json(feature_columns_path)
    if err:
        raise FileNotFoundError(err)
    if not isinstance(payload, list):
        raise ValueError(f"Formato invalido en {feature_columns_path.name}; se esperaba una lista")
    return [str(col) for col in payload]


@lru_cache(maxsize=16)
def load_model_bundle(model_role: str) -> dict[str, Any]:
    config = get_model_role_config(model_role)
    model, err = safe_load_model(Path(config["model_path"]))
    if err:
        raise FileNotFoundError(err)

    feature_columns = load_feature_columns(model_role)
    metadata = load_model_metadata(model_role)
    metrics = metadata.get("test_metrics") or metadata.get("validation_metrics") or {}

    return {
        "model_role": model_role,
        "model": model,
        "feature_columns": feature_columns,
        "uses_target_lags": bool(config["uses_target_lags"]),
        "operational_assumption": str(config["operational_assumption"]),
        "recommended_use": str(config["recommended_use"]),
        "model_path": Path(config["model_path"]),
        "feature_columns_path": Path(config["feature_columns_path"]),
        "metadata_path": Path(config["metadata_path"]),
        "metadata": metadata,
        "metrics": metrics,
    }


def validate_input_features(input_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    expected_columns = list(feature_columns)
    incoming_columns = list(input_df.columns)

    missing_features = [col for col in expected_columns if col not in incoming_columns]
    extra_features = [col for col in incoming_columns if col not in expected_columns]

    aligned_df = input_df.copy()
    if extra_features:
        aligned_df = aligned_df.drop(columns=extra_features, errors="ignore")

    for col in expected_columns:
        if col not in aligned_df.columns:
            aligned_df[col] = pd.NA

    aligned_df = aligned_df[expected_columns]

    non_numeric_features: list[str] = []
    for col in expected_columns:
        converted = pd.to_numeric(aligned_df[col], errors="coerce")
        invalid_mask = aligned_df[col].notna() & converted.isna()
        if bool(invalid_mask.any()):
            non_numeric_features.append(col)
        aligned_df[col] = converted

    null_features = [col for col in expected_columns if bool(aligned_df[col].isna().any())]
    validation_status = not (missing_features or null_features or non_numeric_features)

    return {
        "validation_status": validation_status,
        "missing_features": missing_features,
        "extra_features": extra_features,
        "null_features": null_features,
        "non_numeric_features": non_numeric_features,
        "ordered_df": aligned_df,
    }


def _prediction_warnings(uses_target_lags: bool, validation: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if uses_target_lags:
        warnings.append(
            "Prediction is valid only if lagged lab values are available at inference time."
        )
    if validation.get("extra_features"):
        warnings.append("Extra features were ignored during inference.")
    return warnings


def predict_one(model_role: str, features_dict: dict[str, Any]) -> dict[str, Any]:
    if model_role not in available_model_roles():
        raise KeyError(f"Model role no soportado: {model_role}")
    if not isinstance(features_dict, dict) or not features_dict:
        raise ValueError("features_dict debe ser un diccionario no vacio")

    bundle = load_model_bundle(model_role)
    input_df = pd.DataFrame([features_dict])
    validation = validate_input_features(input_df, bundle["feature_columns"])

    if not validation["validation_status"]:
        raise ValueError("Input invalido; revisar validation")

    ordered_df = validation["ordered_df"]
    prediction = float(bundle["model"].predict(ordered_df)[0])

    return {
        "target": TARGET_NAME,
        "prediction": prediction,
        "model_role": bundle["model_role"],
        "uses_target_lags": bundle["uses_target_lags"],
        "operational_assumption": bundle["operational_assumption"],
        "warnings": _prediction_warnings(bundle["uses_target_lags"], validation),
        "validation": {
            "validation_status": validation["validation_status"],
            "missing_features": validation["missing_features"],
            "extra_features": validation["extra_features"],
            "null_features": validation["null_features"],
            "non_numeric_features": validation["non_numeric_features"],
        },
    }


# Backwards-compatible alias for earlier notebooks/scripts.
predict = predict_one


def summarize_feature_groups(feature_columns: list[str]) -> dict[str, Any]:
    target_lag_features = [col for col in feature_columns if col.startswith(TARGET_LAG_PREFIX)]
    process_features = [
        col
        for col in feature_columns
        if col not in target_lag_features and col not in TEMPORAL_FEATURES
    ]
    return {
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "target_lag_features": target_lag_features,
        "process_features": process_features,
    }
