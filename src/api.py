from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .app_utils import safe_load_csv, safe_load_json
from .config import (
    REPORTS_FIGURES_DIR,
    REPORTS_METRICS_DIR,
    TARGET_NAME,
    available_model_roles,
    get_model_role_config,
    get_project_root,
)
from .inference import load_feature_columns, load_model_bundle, predict_one, summarize_feature_groups
from .simulation import simulate

app = FastAPI(
    title="Minsur Silica Soft Sensor API",
    description="Level 7 API Exposure for inference and predictive sensitivity simulation.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    model_role: str = Field(..., description="Model role for inference")
    features: dict[str, Any] = Field(..., description="Feature payload")


class SimulateRequest(BaseModel):
    model_role: str = Field(..., description="Model role for simulation")
    base_features: dict[str, Any] = Field(..., description="Base feature payload")
    modifications: dict[str, dict[str, Any]] = Field(..., description="Scenario modifications")


def _metadata_found() -> bool:
    return (get_project_root() / "models" / "selected" / "selected_model_metadata.json").exists()


def _load_model_artifact_status() -> dict[str, bool]:
    status: dict[str, bool] = {}
    for role in available_model_roles():
        cfg = get_model_role_config(role)
        status[role] = Path(cfg["model_path"]).exists()
    return status


def _load_feature_file_status() -> dict[str, bool]:
    status: dict[str, bool] = {}
    for role in available_model_roles():
        cfg = get_model_role_config(role)
        status[role] = Path(cfg["feature_columns_path"]).exists()
    return status


def _load_metadata(model_role: str | None = None) -> dict[str, Any]:
    if model_role is None:
        payload, _ = safe_load_json(get_project_root() / "models" / "selected" / "selected_model_metadata.json")
        return payload if isinstance(payload, dict) else {}

    cfg = get_model_role_config(model_role)
    payload, _ = safe_load_json(Path(cfg["metadata_path"]))
    return payload if isinstance(payload, dict) else {}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _metadata_found() else "degraded",
        "metadata_found": _metadata_found(),
        "model_artifacts_found": _load_model_artifact_status(),
        "feature_files_found": _load_feature_file_status(),
        "available_model_roles": available_model_roles(),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    metadata = _load_metadata()
    recommended_role = "recommended_model_with_lagged_lab_assumption"

    role_details: dict[str, Any] = {}
    warnings: list[str] = []

    for role in available_model_roles():
        cfg = get_model_role_config(role)
        role_metadata = _load_metadata(role)
        metrics = role_metadata.get("test_metrics") or role_metadata.get("validation_metrics")
        if not metrics and role in metadata:
            metrics = metadata.get(role, {}).get("test_metrics") or metadata.get(role, {}).get("validation_metrics")

        role_details[role] = {
            "uses_target_lags": bool(cfg["uses_target_lags"]),
            "operational_assumption": cfg["operational_assumption"],
            "recommended_use": cfg["recommended_use"],
            "metrics": metrics or {},
        }

        if cfg["uses_target_lags"]:
            warnings.append(
                f"{role}: Prediction is valid only if lagged lab values are available at inference time."
            )
        else:
            warnings.append(f"{role}: strict no-lab mode; lower predictive strength expected.")

    recommended_metrics = role_details[recommended_role]["metrics"]

    return {
        "target": TARGET_NAME,
        "recommended_model": recommended_role,
        "available_model_roles": available_model_roles(),
        "metrics": recommended_metrics,
        "operational_assumptions": {
            role: details["operational_assumption"] for role, details in role_details.items()
        },
        "warnings": warnings,
    }


@app.get("/features")
def features(model_role: str = Query(..., description="Model role")) -> dict[str, Any]:
    if model_role not in available_model_roles():
        raise HTTPException(status_code=404, detail=f"Model role no encontrado: {model_role}")

    try:
        feature_columns = load_feature_columns(model_role)
        cfg = get_model_role_config(model_role)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudieron cargar features: {exc}") from exc

    grouped = summarize_feature_groups(feature_columns)
    grouped.update(
        {
            "model_role": model_role,
            "uses_target_lags": bool(cfg["uses_target_lags"]),
        }
    )
    return grouped


@app.post("/predict")
def predict_endpoint(payload: PredictRequest) -> dict[str, Any]:
    if payload.model_role not in available_model_roles():
        raise HTTPException(status_code=404, detail=f"Model role no encontrado: {payload.model_role}")

    try:
        return predict_one(payload.model_role, payload.features)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Falla cargando artefactos del modelo: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno en prediccion: {exc}") from exc


@app.post("/simulate")
def simulate_endpoint(payload: SimulateRequest) -> dict[str, Any]:
    if payload.model_role not in available_model_roles():
        raise HTTPException(status_code=404, detail=f"Model role no encontrado: {payload.model_role}")

    try:
        return simulate(payload.model_role, payload.base_features, payload.modifications)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Falla cargando artefactos del modelo: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno en simulacion: {exc}") from exc
