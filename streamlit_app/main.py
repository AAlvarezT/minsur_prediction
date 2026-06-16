from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.app_utils import (
    build_portfolio_row,
    load_or_build_default_row,
    safe_load_csv,
    safe_load_image,
    safe_load_json,
    safe_load_model,
    safe_load_text,
    validate_input_features,
)
from streamlit_app.simulation import apply_modifications

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_NAME = "% Silica Concentrate"

st.set_page_config(
    page_title="Data-Driven Quality System - Silica Concentrate Soft Sensor",
    layout="wide",
)

st.title("Data-Driven Quality System - Silica Concentrate Soft Sensor")
st.caption("Prototipo de soporte a decision para calidad metalurgica (Minsur)")


def _path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def show_optional_dataframe(path: Path, title: str) -> pd.DataFrame | None:
    df, err = safe_load_csv(path)
    st.subheader(title)
    if err:
        st.warning(err)
        return None
    st.dataframe(df, use_container_width=True)
    return df


def show_optional_image(path: Path, caption: str) -> None:
    img, err = safe_load_image(path)
    if err:
        st.warning(err)
        return
    st.image(str(img), caption=caption, use_container_width=True)


@st.cache_data(show_spinner=False)
def load_main_metadata() -> tuple[dict, str | None]:
    return safe_load_json(_path("models", "selected", "selected_model_metadata.json"))


@st.cache_data(show_spinner=False)
def load_high_frequency_metrics() -> tuple[pd.DataFrame | None, str | None]:
    return safe_load_csv(_path("reports", "metrics", "high_frequency_aggregation_comparison.csv"))


def _extract_recommended_metrics(
    metadata: dict,
    hf_df: pd.DataFrame | None,
) -> tuple[float, float, float, str]:
    fallback = metadata.get("recommended_model_with_lagged_lab_assumption", {}).get("test_metrics", {})
    mae = float(fallback.get("MAE", 0.55))
    rmse = float(fallback.get("RMSE", 0.7474))
    r2 = float(fallback.get("R2", 0.6096))
    source = "Modelo recomendado original"

    if hf_df is not None and not hf_df.empty:
        mask = (
            hf_df["scenario"].astype(str).str.strip().eq("lagged_lab_plus_sensor_aggregated")
            & hf_df["model"].astype(str).str.strip().eq("Random Forest")
        )
        matched = hf_df[mask]
        if not matched.empty:
            row = matched.iloc[0]
            mae = float(row.get("test_MAE", mae))
            rmse = float(row.get("test_RMSE", rmse))
            r2 = float(row.get("test_R2", r2))
            source = "Experimento de agregacion intra-horaria"

    return mae, rmse, r2, source


def _build_model_options(metadata: dict) -> dict[str, dict]:
    return {
        "recommended_model_with_lagged_lab_assumption": {
            "model_path": _path("models", "selected", "model_lagged_lab_assumption.pkl"),
            "feature_path": _path("models", "selected", "feature_columns.json"),
            "uses_target_lags": True,
            "assumption": "Valido bajo supuesto de disponibilidad de laboratorio rezagado.",
        },
        "strict_no_lab_input_fallback": {
            "model_path": _path("models", "selected", "model_strict_no_lab_input.pkl"),
            "feature_path": _path("models", "selected", "feature_columns_strict_no_lab_input.json"),
            "uses_target_lags": False,
            "assumption": "Fallback estricto sin entradas de laboratorio recientes.",
        },
        "lag_1_available": {
            "model_path": _path("models", "selected", "lag_1_available_best_model.pkl"),
            "feature_path": _path("models", "selected", "feature_columns.json"),
            "uses_target_lags": True,
            "assumption": "Asume disponibilidad lag_1 (~1 hora).",
        },
        "lag_3_available": {
            "model_path": _path("models", "selected", "lag_3_available_best_model.pkl"),
            "feature_path": _path("models", "selected", "feature_columns.json"),
            "uses_target_lags": True,
            "assumption": "Asume disponibilidad lag_3.",
        },
        "lag_6_available": {
            "model_path": _path("models", "selected", "lag_6_available_best_model.pkl"),
            "feature_path": _path("models", "selected", "feature_columns.json"),
            "uses_target_lags": True,
            "assumption": "Asume disponibilidad lag_6 (senal mas debil).",
        },
        "no_recent_lab_available": {
            "model_path": _path("models", "selected", "no_recent_lab_available_best_model.pkl"),
            "feature_path": _path("models", "selected", "feature_columns_strict_no_lab_input.json"),
            "uses_target_lags": False,
            "assumption": "No hay laboratorio reciente disponible.",
        },
    }


def _get_feature_columns(path: Path) -> tuple[list[str], str | None]:
    payload, err = safe_load_json(path)
    if err:
        return [], err
    if not isinstance(payload, list):
        return [], f"Formato invalido en {path.name}."
    return [str(c) for c in payload], None


def _predict(model, row_df: pd.DataFrame) -> float:
    pred = model.predict(row_df)
    return float(pred[0])


if "last_input_row" not in st.session_state:
    st.session_state["last_input_row"] = None
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "last_model_role" not in st.session_state:
    st.session_state["last_model_role"] = None

section = st.sidebar.radio(
    "Navegacion",
    [
        "Executive Overview",
        "Model Portfolio",
        "Prediction Console",
        "Explainability",
        "What-if Simulator",
        "Laboratory Delay Sensitivity",
        "High-Frequency Sensor Aggregation",
        "MLOps & Reproducibility",
        "Production Readiness",
    ],
)

metadata, metadata_err = load_main_metadata()
hf_df, _ = load_high_frequency_metrics()
if metadata is None:
    metadata = {}

if section == "Executive Overview":
    st.header("Executive Overview")
    st.write(f"Target obligatorio: **{TARGET_NAME}**")

    if metadata_err:
        st.warning(metadata_err)

    mae, rmse, r2, metric_source = _extract_recommended_metrics(metadata, hf_df)
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("R2", f"{r2:.4f}")
    st.caption(f"Fuente de metricas principales: {metric_source}")

    model_name = (
        metadata.get("recommended_model_with_lagged_lab_assumption", {}).get("model_name")
        or "Random Forest"
    )
    st.info(
        "The system is useful under a lagged-laboratory availability assumption. "
        "Sensor-only predictions are available as fallback but have weaker performance."
    )
    st.write(f"Modelo recomendado bajo supuesto lagged-lab: **{model_name}**")

    semaforo = pd.DataFrame(
        {
            "Escenario": ["lag_1_available", "lag_3_available", "lag_6_available", "no_recent_lab_available"],
            "Semaforo": ["GREEN", "YELLOW", "RED", "RED"],
            "Lectura ejecutiva": [
                "Confiable para soporte a decision",
                "Util con cautela",
                "Riesgo alto de perdida de senal",
                "Fallback de baja capacidad explicativa",
            ],
        }
    )
    st.dataframe(semaforo, use_container_width=True)

elif section == "Model Portfolio":
    st.header("Model Portfolio")
    rows: list[dict] = []

    rec = metadata.get("recommended_model_with_lagged_lab_assumption", {})
    rec_test = rec.get("test_metrics", {})
    rows.append(
        build_portfolio_row(
            role="recommended_model_with_lagged_lab_assumption",
            scenario=str(rec.get("scenario", "Feed ON + TargetLags ON")),
            model_name=str(rec.get("model_name", "Random Forest")),
            mae=rec_test.get("MAE"),
            rmse=rec_test.get("RMSE"),
            r2=rec_test.get("R2"),
            uses_target_lags=rec.get("uses_target_lags"),
            operational_assumption="Supuesto de disponibilidad lagged-lab",
            recommended_use="Modelo primario",
        )
    )

    fb = metadata.get("strict_no_lab_input_fallback", {})
    fb_test = fb.get("test_metrics", {})
    rows.append(
        build_portfolio_row(
            role="strict_no_lab_input_fallback",
            scenario=str(fb.get("scenario", "Feed ON + TargetLags OFF")),
            model_name=str(fb.get("model_name", "Random Forest")),
            mae=fb_test.get("MAE"),
            rmse=fb_test.get("RMSE"),
            r2=fb_test.get("R2"),
            uses_target_lags=fb.get("uses_target_lags"),
            operational_assumption="Sin laboratorio reciente",
            recommended_use="Fallback de continuidad",
        )
    )

    lab_df, lab_err = safe_load_csv(_path("reports", "metrics", "lab_delay_availability_scenarios.csv"))
    if lab_df is not None and not lab_df.empty:
        for _, row in lab_df.iterrows():
            rows.append(
                build_portfolio_row(
                    role=str(row.get("Scenario", "")),
                    scenario=str(row.get("Scenario", "")),
                    model_name=str(row.get("Best model", "")),
                    mae=row.get("Test MAE"),
                    rmse=row.get("Test RMSE"),
                    r2=row.get("Test R2"),
                    uses_target_lags=str(row.get("Allowed lags", "none")).lower() != "none",
                    operational_assumption=str(row.get("Availability assumption", "")),
                    recommended_use="Escenario de disponibilidad de laboratorio",
                )
            )
    elif lab_err:
        st.warning(lab_err)

    if hf_df is not None and not hf_df.empty:
        keep = hf_df[hf_df["scenario"].isin(["sensor_only_hourly_aggregated", "lagged_lab_plus_sensor_aggregated"])]
        for _, row in keep.iterrows():
            rows.append(
                build_portfolio_row(
                    role=str(row.get("scenario", "")),
                    scenario=str(row.get("scenario", "")),
                    model_name=str(row.get("model", "")),
                    mae=row.get("test_MAE"),
                    rmse=row.get("test_RMSE"),
                    r2=row.get("test_R2"),
                    uses_target_lags="lagged_lab" in str(row.get("scenario", "")),
                    operational_assumption="Experimento de agregacion intra-horaria",
                    recommended_use="Benchmark/extension",
                )
            )

    portfolio_df = pd.DataFrame(rows)
    st.dataframe(portfolio_df, use_container_width=True)

    show_optional_dataframe(_path("reports", "metrics", "model_version_audit.csv"), "Model version audit")
    show_optional_dataframe(
        _path("reports", "metrics", "high_frequency_aggregation_comparison.csv"),
        "High-frequency aggregation comparison",
    )
    show_optional_dataframe(
        _path("reports", "metrics", "lab_delay_availability_scenarios.csv"),
        "Lab delay availability scenarios",
    )

elif section == "Prediction Console":
    st.header("Prediction Console")

    model_options = _build_model_options(metadata)
    role = st.selectbox("Selecciona modelo", list(model_options.keys()))
    model_cfg = model_options[role]

    model, model_err = safe_load_model(model_cfg["model_path"])
    if model_err:
        st.warning(model_err)

    feature_columns, feature_err = _get_feature_columns(model_cfg["feature_path"])
    if feature_err:
        st.warning(feature_err)

    if model is None or not feature_columns:
        st.stop()

    input_mode = st.radio(
        "Fuente de features",
        [
            "Cargar CSV (1 fila)",
            "Usar ejemplo precargado",
            "Usar defaults (mediana si existe dataset procesado)",
        ],
    )

    input_df: pd.DataFrame | None = None

    if input_mode == "Cargar CSV (1 fila)":
        uploaded = st.file_uploader("Sube un CSV con 1 fila", type=["csv"])
        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)
                if input_df.shape[0] > 1:
                    st.warning("Se detectaron varias filas; se usara solo la primera.")
                    input_df = input_df.head(1)
            except Exception as exc:
                st.error(f"No se pudo leer el CSV: {exc}")

    if input_mode in [
        "Usar ejemplo precargado",
        "Usar defaults (mediana si existe dataset procesado)",
    ]:
        example_df, origin_note = load_or_build_default_row(PROJECT_ROOT, feature_columns)
        st.info(origin_note)
        input_df = example_df

    if input_df is not None:
        st.write("Vista previa de entrada")
        st.dataframe(input_df.iloc[:, : min(12, input_df.shape[1])], use_container_width=True)

        report = validate_input_features(input_df, feature_columns)

        if report["missing_features"]:
            st.error(f"Faltan features requeridas ({len(report['missing_features'])}).")
            st.write(report["missing_features"][:20])
        if report["extra_features"]:
            st.warning("Hay columnas extra. Se ignoraran en prediccion.")
            st.write(report["extra_features"][:20])
        if report["non_numeric_features"]:
            st.error("Hay features no numericas.")
            st.write(report["non_numeric_features"][:20])
        if report["null_features"]:
            st.error("Hay features con nulos.")
            st.write(report["null_features"][:20])

        if st.button("Predecir", type="primary"):
            if not report["validation_status"]:
                st.error("Validacion fallida: no se puede predecir.")
            else:
                ordered_df = report["ordered_df"]
                pred = _predict(model, ordered_df)
                st.session_state["last_input_row"] = ordered_df
                st.session_state["last_prediction"] = pred
                st.session_state["last_model_role"] = role

                st.success(f"Predicted {TARGET_NAME}: {pred:.4f}")
                st.write(f"model role: {role}")
                st.write(f"Depende de target lags: {model_cfg['uses_target_lags']}")
                if model_cfg["uses_target_lags"]:
                    st.warning("Valid only if lagged lab values are available at inference time.")
                st.info(model_cfg["assumption"])

elif section == "Explainability":
    st.header("Explainability")
    st.info("Insight principal: % Silica Concentrate_lag_1 domina la senal predictiva.")
    st.write(
        "Interpretacion ejecutiva: persistencia temporal, supuesto de disponibilidad operativa y lectura no causal."
    )

    show_optional_image(_path("reports", "figures", "shap_bar.png"), "SHAP global bar")
    show_optional_image(_path("reports", "figures", "shap_summary.png"), "SHAP summary")
    show_optional_dataframe(_path("reports", "metrics", "top_shap_features.csv"), "Top SHAP features")
    show_optional_dataframe(_path("reports", "metrics", "local_explanations.csv"), "Local explanations (precalculadas)")

    st.subheader("Drivers locales de la prediccion actual")
    latest_row = st.session_state.get("last_input_row")
    latest_role = st.session_state.get("last_model_role")
    if latest_row is None or latest_role is None:
        st.info("Primero ejecuta una prediccion en Prediction Console para ver drivers locales.")
    else:
        model_cfg = _build_model_options(metadata).get(latest_role)
        model, model_err = safe_load_model(model_cfg["model_path"])
        if model_err:
            st.warning(model_err)

        if model is not None:
            use_shap = st.checkbox("Intentar SHAP local (top 5 +/-)", value=True)
            local_done = False
            if use_shap:
                try:
                    import shap

                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(latest_row)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    vals = np.array(shap_values).reshape(-1)
                    local_df = pd.DataFrame(
                        {
                            "feature": latest_row.columns,
                            "shap_value": vals,
                            "feature_value": latest_row.iloc[0].values,
                        }
                    )
                    top_pos = local_df.sort_values("shap_value", ascending=False).head(5)
                    top_neg = local_df.sort_values("shap_value", ascending=True).head(5)

                    c1, c2 = st.columns(2)
                    c1.write("Top positive drivers")
                    c1.dataframe(top_pos, use_container_width=True)
                    c2.write("Top negative drivers")
                    c2.dataframe(top_neg, use_container_width=True)
                    local_done = True
                except Exception as exc:
                    st.warning(f"No se pudo calcular SHAP local en linea: {exc}")

            if not local_done:
                top_shap, top_err = safe_load_csv(_path("reports", "metrics", "top_shap_features.csv"))
                if top_err:
                    st.warning(top_err)
                else:
                    st.write("Fallback: top global features")
                    st.dataframe(top_shap.head(10), use_container_width=True)

    st.warning("No usar lenguaje causal: esta seccion refleja sensibilidad del modelo.")

elif section == "What-if Simulator":
    st.header("What-if Simulator")
    st.warning("These simulations are predictive sensitivities, not causal recommendations.")

    model_options = _build_model_options(metadata)
    sim_role = st.selectbox("Modelo para simulacion", list(model_options.keys()), index=0)
    sim_model_cfg = model_options[sim_role]

    model, model_err = safe_load_model(sim_model_cfg["model_path"])
    if model_err:
        st.warning(model_err)

    feature_columns, feature_err = _get_feature_columns(sim_model_cfg["feature_path"])
    if feature_err:
        st.warning(feature_err)

    if model is None or not feature_columns:
        st.stop()

    if st.session_state.get("last_input_row") is not None:
        base_row = st.session_state["last_input_row"].copy()
        st.info("Base tomada desde la ultima prediccion de la sesion.")
    else:
        base_row, origin_note = load_or_build_default_row(PROJECT_ROOT, feature_columns)
        st.info(origin_note)

    controls = [
        ("Amina Flow", "pct", st.slider("Amina Flow %", -20.0, 20.0, 0.0, 0.5)),
        ("Starch Flow", "pct", st.slider("Starch Flow %", -20.0, 20.0, 0.0, 0.5)),
        ("Ore Pulp pH", "abs_delta", st.slider("Ore Pulp pH delta", -1.0, 1.0, 0.0, 0.05)),
        ("Ore Pulp Density", "pct", st.slider("Ore Pulp Density %", -20.0, 20.0, 0.0, 0.5)),
        (
            "Flotation Column 01 Level",
            "pct",
            st.slider("Flotation Column Level % (columna 01)", -20.0, 20.0, 0.0, 0.5),
        ),
        (
            "Flotation Column 01 Air Flow",
            "pct",
            st.slider("Flotation Column Air Flow % (columna 01)", -20.0, 20.0, 0.0, 0.5),
        ),
        (
            "% Silica Concentrate_lag_1",
            "abs_delta",
            st.slider("recent lab trend delta (lag_1)", -1.0, 1.0, 0.0, 0.05),
        ),
    ]

    modifications = {
        feat: {"mode": mode, "value": val}
        for feat, mode, val in controls
        if abs(float(val)) > 1e-12
    }

    if st.button("Simular escenario", type="primary"):
        scenario_row, warnings, applied = apply_modifications(base_row, modifications)
        base_pred = _predict(model, base_row)
        scn_pred = _predict(model, scenario_row)
        delta = scn_pred - base_pred
        delta_pct = (delta / base_pred * 100.0) if base_pred != 0 else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("base_prediction", f"{base_pred:.4f}")
        c2.metric("scenario_prediction", f"{scn_pred:.4f}")
        c3.metric("delta_prediction", f"{delta:.4f}")
        c4.metric("delta_prediction_pct", f"{delta_pct:.2f}%")

        if applied:
            st.write("modified features")
            st.dataframe(pd.DataFrame(applied), use_container_width=True)
        else:
            st.info("No se aplicaron cambios (todos los deltas en 0).")

        for w in warnings:
            st.warning(w)

    show_optional_dataframe(_path("reports", "metrics", "scenario_results.csv"), "Scenario results (precalculados)")
    show_optional_dataframe(_path("reports", "metrics", "scenario_ranking.csv"), "Scenario ranking (precalculado)")
    show_optional_image(_path("reports", "figures", "scenario_impact_heatmap.png"), "Scenario impact heatmap")

elif section == "Laboratory Delay Sensitivity":
    st.header("Laboratory Delay Sensitivity")
    lab_df, lab_err = safe_load_csv(_path("reports", "metrics", "lab_delay_availability_scenarios.csv"))
    if lab_err:
        st.warning(lab_err)
    else:
        st.dataframe(lab_df, use_container_width=True)

        if {"Scenario", "Validation R2"}.issubset(set(lab_df.columns)):
            fig = px.bar(
                lab_df,
                x="Scenario",
                y="Validation R2",
                title="Validation R2 by laboratory availability",
            )
            st.plotly_chart(fig, use_container_width=True)

    show_optional_image(_path("reports", "figures", "lab_delay_availability_r2.png"), "R2 por disponibilidad de laboratorio")
    st.warning("Model value depends critically on the latest available laboratory result.")
    st.write(
        "Lectura operativa: lag_1 equivale aproximadamente a 1 hora. "
        "Si el laboratorio demora mas, el sistema debe migrar a escenarios lag_3/lag_6 o fallback sin laboratorio."
    )

elif section == "High-Frequency Sensor Aggregation":
    st.header("High-Frequency Sensor Aggregation")

    summary = pd.DataFrame(
        {
            "Caso": [
                "original sensor-only",
                "sensor-only aggregated",
                "persistence baseline",
                "lagged lab + sensor aggregated",
            ],
            "R2 aprox": [0.0480, 0.1260, 0.5965, 0.6190],
        }
    )
    st.dataframe(summary, use_container_width=True)

    hf_comp, hf_err = safe_load_csv(_path("reports", "metrics", "high_frequency_aggregation_comparison.csv"))
    if hf_err:
        st.warning(hf_err)
    else:
        st.dataframe(hf_comp, use_container_width=True)

    show_optional_image(
        _path("reports", "figures", "high_frequency_aggregation_r2_comparison.png"),
        "Comparacion R2",
    )
    show_optional_image(
        _path("reports", "figures", "high_frequency_aggregation_mae_comparison.png"),
        "Comparacion MAE",
    )
    show_optional_image(
        _path("reports", "figures", "sensor_only_aggregated_feature_importance.png"),
        "Feature importance sensor-only aggregated",
    )

    st.info(
        "Conclusion: la agregacion intra-horaria recupera parte de la senal de sensores. "
        "Aun asi, la persistencia temporal asociada al historial de laboratorio sigue dominando la capacidad predictiva."
    )

elif section == "MLOps & Reproducibility":
    st.header("MLOps & Reproducibility")

    st.subheader("Metadata del modelo seleccionado")
    if metadata_err:
        st.warning(metadata_err)
    else:
        st.json(metadata)

    show_optional_dataframe(_path("reports", "metrics", "model_version_audit.csv"), "Model version audit")
    show_optional_dataframe(
        _path("reports", "metrics", "reproducibility_artifact_checklist.csv"),
        "Reproducibility artifact checklist",
    )
    show_optional_dataframe(_path("reports", "metrics", "mlflow_runs_audit.csv"), "MLflow runs audit")
    show_optional_dataframe(
        _path("reports", "metrics", "mlflow_leaderboard_audit.csv"),
        "MLflow leaderboard audit",
    )

    st.code(
        "set MLFLOW_ALLOW_FILE_STORE=true && python -m mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5001",
        language="bash",
    )

    mlops_summary, mlops_err = safe_load_text(_path("reports", "metrics", "mlops_summary.md"))
    st.subheader("MLOps summary")
    if mlops_err:
        st.warning(mlops_err)
    else:
        st.markdown(mlops_summary)

    st.info(
        "Estado actual: tracking local con MLflow + respaldo CSV/JSON para reproducibilidad. "
        "Siguiente paso: Model Registry formal y flujo de deployment monitoreado."
    )
    st.warning("No hay deployment productivo implementado aun.")

elif section == "Production Readiness":
    st.header("Production Readiness")

    checklist = pd.DataFrame(
        {
            "Item": [
                "temporal validation",
                "no contemporaneous target leakage",
                "explainability available",
                "model metadata available",
                "feature list available",
                "fallback model available",
                "what-if available",
                "lab delay still needs validation",
                "drift monitoring not implemented",
                "formal model registry not implemented",
            ],
            "Status": [
                "done",
                "done",
                "done",
                "done",
                "done",
                "done",
                "done",
                "pending",
                "pending",
                "pending",
            ],
        }
    )
    st.dataframe(checklist, use_container_width=True)

    st.subheader("Roadmap")
    roadmap = [
        "validate lab delay with operations",
        "validate feed variable availability",
        "formalize MLflow registry",
        "implement API/FastAPI or batch scoring",
        "implement drift monitoring",
        "validate what-if scenarios with metallurgists",
    ]
    for step in roadmap:
        st.write(f"- {step}")

    show_optional_dataframe(
        _path("reports", "metrics", "production_readiness_roadmap.csv"),
        "Roadmap audit table",
    )

st.sidebar.info(
    "Lenguaje de interpretacion: senal predictiva, sensibilidad del modelo y soporte a decision."
)
