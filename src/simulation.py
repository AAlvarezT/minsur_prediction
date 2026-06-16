from __future__ import annotations

from typing import Any

import pandas as pd

from .inference import TARGET_NAME, load_model_bundle, predict_one, validate_input_features

LAB_TREND_WARNING = "This is a lab trend scenario, not an operational lever."
SIMULATION_DISCLAIMER = "Predictive sensitivity only; not a causal recommendation."
TARGET_LAG_PREFIX = "% Silica Concentrate_lag_"


def apply_modifications(
    base_row: pd.DataFrame,
    modifications: dict[str, dict[str, float | int | str]],
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    if base_row.shape[0] != 1:
        raise ValueError("base_row debe tener exactamente una fila")

    scenario = base_row.copy()
    warnings: list[str] = []
    modified_features: list[dict[str, Any]] = []

    for feature, spec in modifications.items():
        if feature not in scenario.columns:
            warnings.append(f"Feature inexistente omitida: {feature}")
            continue

        mode = str(spec.get("mode", "")).strip().lower()
        value = float(spec.get("value", 0.0))
        old_value = float(scenario.iloc[0][feature])

        if mode == "pct":
            new_value = old_value * (1 + value / 100.0)
        elif mode == "abs_delta":
            new_value = old_value + value
        else:
            warnings.append(f"Modo no soportado para {feature}: {mode}")
            continue

        scenario.at[scenario.index[0], feature] = new_value
        modified_features.append(
            {
                "feature": feature,
                "mode": mode,
                "value": value,
                "old_value": old_value,
                "new_value": float(new_value),
            }
        )

        if feature.startswith(TARGET_LAG_PREFIX):
            warnings.append(LAB_TREND_WARNING)

    return scenario, warnings, modified_features


def simulate(
    model_role: str,
    base_features: dict[str, Any],
    modifications: dict[str, dict[str, float | int | str]],
) -> dict[str, Any]:
    if not isinstance(base_features, dict) or not base_features:
        raise ValueError("base_features debe ser un diccionario no vacio")
    if not isinstance(modifications, dict):
        raise ValueError("modifications debe ser un diccionario")

    bundle = load_model_bundle(model_role)
    base_df = pd.DataFrame([base_features])
    base_validation = validate_input_features(base_df, bundle["feature_columns"])

    if not base_validation["validation_status"]:
        raise ValueError("Base features invalidas; revisar validation")

    base_row = base_validation["ordered_df"]
    scenario_row, mod_warnings, modified_features = apply_modifications(base_row, modifications)

    base_result = predict_one(model_role, base_row.iloc[0].to_dict())
    scenario_result = predict_one(model_role, scenario_row.iloc[0].to_dict())

    base_prediction = float(base_result["prediction"])
    scenario_prediction = float(scenario_result["prediction"])
    delta_prediction = scenario_prediction - base_prediction
    delta_prediction_pct = (delta_prediction / base_prediction * 100.0) if base_prediction != 0 else 0.0

    warnings = list(
        dict.fromkeys(
            list(base_result.get("warnings", []))
            + list(scenario_result.get("warnings", []))
            + mod_warnings
        )
    )

    return {
        "target": TARGET_NAME,
        "base_prediction": base_prediction,
        "scenario_prediction": scenario_prediction,
        "delta_prediction": delta_prediction,
        "delta_prediction_pct": delta_prediction_pct,
        "modified_features": modified_features,
        "warnings": warnings,
        "disclaimer": SIMULATION_DISCLAIMER,
        "validation": base_validation,
    }
