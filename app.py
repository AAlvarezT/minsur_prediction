from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.app_utils import safe_load_csv, safe_load_image, safe_load_json, safe_load_text
from src.config import (
    REPORTS_FIGURES_DIR,
    REPORTS_METRICS_DIR,
    TARGET_NAME,
    available_model_roles,
    get_model_role_config,
    get_project_root,
)
from src.inference import load_feature_columns, load_model_bundle, predict_one
from src.simulation import simulate

st.set_page_config(page_title="Minsur Level 7 API Demo", layout="wide")
st.title("Minsur Level 7 - API Exposure Demo")
st.caption("Executive demo for inference, sensitivity, and MLOps evidence.")


def _optional_df(filename: str) -> pd.DataFrame | None:
    df, err = safe_load_csv(REPORTS_METRICS_DIR / filename)
    if err:
        st.warning(err)
        return None
    return df


@st.cache_data(show_spinner=False)
def _model_metadata(role: str | None = None) -> dict[str, Any]:
    if role is None:
        payload, _ = safe_load_json(get_project_root() / "models" / "selected" / "selected_model_metadata.json")
        return payload if isinstance(payload, dict) else {}
    cfg = get_model_role_config(role)
    payload, _ = safe_load_json(Path(cfg["metadata_path"]))
    return payload if isinstance(payload, dict) else {}


@st.cache_data(show_spinner=False)
def _feature_defaults(role: str) -> dict[str, Any]:
    try:
        feature_columns = load_feature_columns(role)
    except Exception:
        return {}

    processed_dir = get_project_root() / "data" / "processed"
    candidates = sorted(processed_dir.glob("*.csv")) + sorted(processed_dir.glob("*.parquet"))
    for file_path in candidates:
        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_parquet(file_path)
            if df.empty:
                continue
            numeric = df.apply(pd.to_numeric, errors="coerce")
            medians = numeric.median(numeric_only=True)
            return {col: float(medians.get(col, 0.0)) for col in feature_columns}
        except Exception:
            continue
    return {col: 0.0 for col in feature_columns}


def _json_to_dict(text_value: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(text_value)
        if not isinstance(payload, dict):
            return None, "El JSON debe ser un objeto tipo diccionario."
        return payload, None
    except Exception as exc:
        return None, f"JSON invalido: {exc}"


def _show_image(name: str, caption: str) -> None:
    path = REPORTS_FIGURES_DIR / name
    image, err = safe_load_image(path)
    if err:
        st.warning(err)
    else:
        st.image(str(image), caption=caption, use_container_width=True)


model_metadata = _model_metadata()
roles = available_model_roles()

section = st.sidebar.radio(
    "Navigation",
    ["Overview", "Model Portfolio", "Prediction Console", "What-if Simulator", "Explainability & MLOps artifacts"],
)

if section == "Overview":
    st.subheader("Overview")
    rec = model_metadata.get("recommended_model_with_lagged_lab_assumption", {})
    rec_test = rec.get("test_metrics", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f"{float(rec_test.get('MAE', 0.5500)):.4f}")
    c2.metric("Test RMSE", f"{float(rec_test.get('RMSE', 0.7474)):.4f}")
    c3.metric("Test R2", f"{float(rec_test.get('R2', 0.6096)):.4f}")

    st.info(
        "The recommended model is valid under a lagged-lab availability assumption. "
        "Sensor-only fallback is available but materially weaker."
    )
    st.write(f"Target: {TARGET_NAME}")

elif section == "Model Portfolio":
    st.subheader("Model Portfolio")
    rows = []
    for role in roles:
        cfg = get_model_role_config(role)
        meta = _model_metadata(role)
        metrics = meta.get("test_metrics") or meta.get("validation_metrics") or {}
        rows.append(
            {
                "model_role": role,
                "recommended_use": cfg["recommended_use"],
                "uses_target_lags": cfg["uses_target_lags"],
                "operational_assumption": cfg["operational_assumption"],
                "MAE": metrics.get("MAE"),
                "RMSE": metrics.get("RMSE"),
                "R2": metrics.get("R2"),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    for filename in [
        "model_version_audit.csv",
        "mlflow_runs_audit.csv",
        "reproducibility_artifact_checklist.csv",
        "high_frequency_aggregation_comparison.csv",
        "lab_delay_availability_scenarios.csv",
    ]:
        st.markdown(f"**{filename}**")
        df = _optional_df(filename)
        if df is not None:
            st.dataframe(df, use_container_width=True)

elif section == "Prediction Console":
    st.subheader("Prediction Console")
    role = st.selectbox("model_role", roles)
    defaults = _feature_defaults(role)
    feature_text = st.text_area(
        "features JSON",
        value=json.dumps(defaults, indent=2, ensure_ascii=True),
        height=280,
    )

    if st.button("Predict", type="primary"):
        payload, err = _json_to_dict(feature_text)
        if err:
            st.error(err)
        else:
            try:
                result = predict_one(role, payload)
                st.success(f"Prediction: {result['prediction']:.4f}")
                st.json(result)
            except Exception as exc:
                st.error(str(exc))

elif section == "What-if Simulator":
    st.subheader("What-if Simulator")
    role = st.selectbox("model_role", roles, key="sim_role")
    defaults = _feature_defaults(role)
    mod_defaults = {
        "Amina Flow": {"mode": "pct", "value": 5},
        "Ore Pulp pH": {"mode": "abs_delta", "value": 0.2},
    }

    base_text = st.text_area("base_features JSON", value=json.dumps(defaults, indent=2, ensure_ascii=True), height=240)
    mod_text = st.text_area("modifications JSON", value=json.dumps(mod_defaults, indent=2, ensure_ascii=True), height=180)

    if st.button("Simulate", type="primary"):
        base_payload, base_err = _json_to_dict(base_text)
        mod_payload, mod_err = _json_to_dict(mod_text)
        if base_err:
            st.error(base_err)
        elif mod_err:
            st.error(mod_err)
        else:
            try:
                result = simulate(role, base_payload, mod_payload)
                st.success("Simulation completed")
                st.json(result)
            except Exception as exc:
                st.error(str(exc))

elif section == "Explainability & MLOps artifacts":
    st.subheader("Explainability & MLOps artifacts")
    st.info(
        "This demo focuses on evidence browsing and inference contracts. "
        "It does not recalculate SHAP or retrain any model."
    )

    st.markdown("**Main evidence**")
    _show_image("shap_bar.png", "Global SHAP bar")
    _show_image("lab_delay_availability_r2.png", "Laboratory delay sensitivity")
    _show_image("scenario_impact_heatmap.png", "Scenario impact heatmap")
    _show_image("high_frequency_aggregation_r2_comparison.png", "High-frequency aggregation R2")
    _show_image("sensor_only_aggregated_feature_importance.png", "Sensor-only aggregated importance")

    for filename in [
        "selected_model_metadata_audit.csv",
        "top_shap_features.csv",
        "local_explanations.csv",
        "scenario_results.csv",
        "scenario_ranking.csv",
        "mlops_summary.md",
    ]:
        st.markdown(f"**{filename}**")
        if filename.endswith(".md"):
            text, err = safe_load_text(REPORTS_METRICS_DIR / filename)
            if err:
                st.warning(err)
            else:
                st.markdown(text)
        else:
            df = _optional_df(filename)
            if df is not None:
                st.dataframe(df, use_container_width=True)

    st.warning(
        "What-if outputs are predictive sensitivities, not causal recommendations or automatic control."
    )
