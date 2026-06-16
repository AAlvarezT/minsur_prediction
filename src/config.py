"""Project configuration and artifact paths.

This module keeps the legacy ``CFG`` singleton for the existing notebooks and
training utilities, while also exposing a stricter API-first configuration used
by inference, simulation, and the Level 7 API exposure layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
TARGET_NAME = "% Silica Concentrate"


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_config(config_path: Path | None = None) -> dict:
    """Load ``config.yaml`` and resolve relative paths to absolute paths."""
    config_path = config_path or CONFIG_PATH

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, rel_path in cfg.get("paths", {}).items():
        abs_path = PROJECT_ROOT / rel_path
        cfg["paths"][key] = abs_path
        abs_path.mkdir(parents=True, exist_ok=True)

    return cfg


CFG = load_config()

MODELS_SELECTED_DIR = CFG["paths"]["models_selected"]
REPORTS_METRICS_DIR = CFG["paths"]["reports_metrics"]
REPORTS_FIGURES_DIR = CFG["paths"]["reports_figures"]


def _role(
    *,
    model_filename: str,
    feature_filename: str,
    metadata_filename: str,
    uses_target_lags: bool,
    operational_assumption: str,
    recommended_use: str,
) -> dict[str, Any]:
    return {
        "model_path": MODELS_SELECTED_DIR / model_filename,
        "feature_columns_path": MODELS_SELECTED_DIR / feature_filename,
        "metadata_path": MODELS_SELECTED_DIR / metadata_filename,
        "uses_target_lags": uses_target_lags,
        "operational_assumption": operational_assumption,
        "recommended_use": recommended_use,
    }


MODEL_ROLE_CONFIGS: dict[str, dict[str, Any]] = {
    "recommended_model_with_lagged_lab_assumption": _role(
        model_filename="model_lagged_lab_assumption.pkl",
        feature_filename="feature_columns.json",
        metadata_filename="selected_model_metadata.json",
        uses_target_lags=True,
        operational_assumption=(
            "Recommended model under lagged-lab availability assumption; valid only if the latest lab result is available before inference."
        ),
        recommended_use="Primary soft sensor for decision support when lagged lab is available.",
    ),
    "strict_no_lab_input_fallback": _role(
        model_filename="model_strict_no_lab_input.pkl",
        feature_filename="feature_columns_strict_no_lab_input.json",
        metadata_filename="selected_model_metadata.json",
        uses_target_lags=False,
        operational_assumption=(
            "Strict fallback without recent laboratory target inputs; lower predictive strength expected."
        ),
        recommended_use="Conservative fallback when no recent lab values are available.",
    ),
    "lag_1_available": _role(
        model_filename="lag_1_available_best_model.pkl",
        feature_filename="feature_columns.json",
        metadata_filename="lag_1_available_best_model_metadata.json",
        uses_target_lags=True,
        operational_assumption="Lag_1 availability assumption; roughly one hour of delay in the hourly table.",
        recommended_use="Best when lag_1 lab availability is operationally valid.",
    ),
    "lag_3_available": _role(
        model_filename="lag_3_available_best_model.pkl",
        feature_filename="feature_columns.json",
        metadata_filename="lag_3_available_best_model_metadata.json",
        uses_target_lags=True,
        operational_assumption="Lag_3 availability assumption; older lab history only.",
        recommended_use="Use when the newest lab value is not yet available but lag_3 exists.",
    ),
    "lag_6_available": _role(
        model_filename="lag_6_available_best_model.pkl",
        feature_filename="feature_columns.json",
        metadata_filename="lag_6_available_best_model_metadata.json",
        uses_target_lags=True,
        operational_assumption="Lag_6 availability assumption; weaker predictive signal.",
        recommended_use="Fallback within delayed-lab scenarios with limited recent information.",
    ),
    "no_recent_lab_available": _role(
        model_filename="no_recent_lab_available_best_model.pkl",
        feature_filename="feature_columns_strict_no_lab_input.json",
        metadata_filename="no_recent_lab_available_best_model_metadata.json",
        uses_target_lags=False,
        operational_assumption="No recent lab values available; sensor-only fallback only.",
        recommended_use="Use when no recent lab measurement can be trusted at inference time.",
    ),
}


def get_model_role_config(model_role: str) -> dict[str, Any]:
    if model_role not in MODEL_ROLE_CONFIGS:
        raise KeyError(f"Model role no soportado: {model_role}")
    return MODEL_ROLE_CONFIGS[model_role]


def available_model_roles() -> list[str]:
    return list(MODEL_ROLE_CONFIGS.keys())
