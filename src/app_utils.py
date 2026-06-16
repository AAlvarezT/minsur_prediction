from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def safe_load_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, f"JSON no encontrado: {path}"
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"Error leyendo JSON {path.name}: {exc}"


def safe_load_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"CSV no encontrado: {path}"
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return None, f"Error leyendo CSV {path.name}: {exc}"


def safe_load_text(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, f"Texto no encontrado: {path}"
    try:
        return path.read_text(encoding="utf-8"), None
    except Exception as exc:
        return None, f"Error leyendo texto {path.name}: {exc}"


def safe_load_image(path: Path) -> tuple[Path | None, str | None]:
    if not path.exists():
        return None, f"Imagen no encontrada: {path}"
    return path, None


def safe_load_model(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, f"Modelo no encontrado: {path}"
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, f"Error cargando modelo {path.name}: {exc}"


def read_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return None
