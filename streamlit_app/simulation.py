from __future__ import annotations

from typing import Any

import pandas as pd


LAG_PREFIX = "% Silica Concentrate_lag_"


def apply_modifications(
    base_row: pd.DataFrame,
    modifications: dict[str, dict[str, float | int | str]],
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    """
    Apply scenario modifications to a single-row dataframe.

    Supported modes:
    - pct: new = old * (1 + value/100)
    - abs_delta: new = old + value
    """
    scenario = base_row.copy()
    warnings: list[str] = []
    applied: list[dict[str, Any]] = []

    if scenario.shape[0] != 1:
        raise ValueError("base_row debe contener exactamente una fila")

    for feature, spec in modifications.items():
        if feature not in scenario.columns:
            warnings.append(f"Feature inexistente omitida: {feature}")
            continue

        mode = str(spec.get("mode", "")).strip().lower()
        value = float(spec.get("value", 0.0))

        old_value = float(scenario.at[scenario.index[0], feature])

        if mode == "pct":
            new_value = old_value * (1.0 + value / 100.0)
        elif mode == "abs_delta":
            new_value = old_value + value
        else:
            warnings.append(f"Modo no soportado para {feature}: {mode}")
            continue

        scenario.at[scenario.index[0], feature] = new_value
        applied.append(
            {
                "feature": feature,
                "old_value": old_value,
                "new_value": float(new_value),
                "mode": mode,
                "value": value,
            }
        )

        if feature.startswith(LAG_PREFIX):
            warnings.append(
                "This is a lab trend scenario, not an operational lever."
            )

    return scenario, warnings, applied
