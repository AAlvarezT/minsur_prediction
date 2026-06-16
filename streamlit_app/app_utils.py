from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def resolve_path(project_root: Path, path_like: str | Path) -> Path:
    """Resolve absolute/relative paths safely from project root."""
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def safe_load_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, f"No se encontro el archivo: {path}"
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:
        return None, f"Error cargando JSON ({path.name}): {exc}"


def safe_load_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"No se encontro el archivo: {path}"
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return None, f"Error cargando CSV ({path.name}): {exc}"


def safe_load_image(path: Path) -> tuple[Path | None, str | None]:
    if not path.exists():
        return None, f"No se encontro la imagen: {path}"
    return path, None


def safe_load_model(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, f"No se encontro el modelo: {path}"
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, f"Error cargando modelo ({path.name}): {exc}"


def safe_load_text(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, f"No se encontro el archivo: {path}"
    try:
        return path.read_text(encoding="utf-8"), None
    except Exception as exc:
        return None, f"Error leyendo texto ({path.name}): {exc}"


def _read_any_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    return None


def load_or_build_default_row(project_root: Path, feature_columns: list[str]) -> tuple[pd.DataFrame, str]:
    """
    Build a single-row input example.

    Priority:
    1) Processed dataset if available (median values).
    2) Zero/default template.
    """
    processed = project_root / "data" / "processed"
    candidates = []
    if processed.exists():
        candidates.extend(sorted(processed.glob("*.parquet")))
        candidates.extend(sorted(processed.glob("*.csv")))

    for path in candidates:
        try:
            df = _read_any_table(path)
            if df is None or df.empty:
                continue
            numeric_df = df.copy()
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
            medians = numeric_df.median(numeric_only=True)
            row = {col: float(medians.get(col, np.nan)) for col in feature_columns}
            out = pd.DataFrame([row])
            out = out.fillna(0.0)
            _inject_temporal_defaults(out)
            return out, f"Ejemplo construido con medianas desde: {path.name}"
        except Exception:
            continue

    out = pd.DataFrame([{col: 0.0 for col in feature_columns}])
    _inject_temporal_defaults(out)
    return out, "No hubo dataset procesado disponible. Se uso plantilla default (0.0 + temporales)."


def _inject_temporal_defaults(df: pd.DataFrame) -> None:
    defaults = {
        "hour": 12.0,
        "day": 15.0,
        "month": 8.0,
        "dayofweek": 2.0,
        "is_weekend": 0.0,
    }
    for col, val in defaults.items():
        if col in df.columns:
            df[col] = val


def validate_input_features(input_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    missing_features = [c for c in feature_columns if c not in input_df.columns]
    extra_features = [c for c in input_df.columns if c not in feature_columns]

    working = input_df.copy()
    if extra_features:
        working = working.drop(columns=extra_features, errors="ignore")

    for col in feature_columns:
        if col not in working.columns:
            working[col] = np.nan

    working = working[feature_columns]

    non_numeric_features: list[str] = []
    for col in feature_columns:
        converted = pd.to_numeric(working[col], errors="coerce")
        bad_mask = converted.isna() & working[col].notna()
        if bool(bad_mask.any()):
            non_numeric_features.append(col)
        working[col] = converted

    null_features = [c for c in feature_columns if working[c].isna().any()]

    validation_status = not (missing_features or null_features or non_numeric_features)

    return {
        "missing_features": missing_features,
        "extra_features": extra_features,
        "null_features": null_features,
        "non_numeric_features": non_numeric_features,
        "validation_status": validation_status,
        "ordered_df": working,
    }


def build_portfolio_row(
    role: str,
    scenario: str,
    model_name: str,
    mae: float | None,
    rmse: float | None,
    r2: float | None,
    uses_target_lags: bool | None,
    operational_assumption: str,
    recommended_use: str,
) -> dict[str, Any]:
    return {
        "model role": role,
        "scenario": scenario,
        "model name": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "uses target lags": uses_target_lags,
        "operational assumption": operational_assumption,
        "recommended use": recommended_use,
    }
