from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
REPORT_CSV = REPORTS_METRICS_DIR / "project_audit_report.csv"
SUMMARY_MD = REPORTS_METRICS_DIR / "project_audit_summary.md"


ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:\\\\Users\\\\"),
    re.compile(r"/Users/"),
    re.compile(r"file:///"),
]


@dataclass(frozen=True)
class ArtifactCheck:
    category: str
    artifact: str
    expected_path: str
    critical: bool
    optional: bool = False
    notes: str = ""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def path_exists(path_str: str) -> bool:
    return (PROJECT_ROOT / path_str).exists()


def notebook_quality_audit(notebook_path: Path) -> dict[str, Any]:
    if not notebook_path.exists():
        return {
            "exists": False,
            "has_intro": False,
            "has_abs_paths": False,
            "error_outputs": 0,
            "large_outputs": 0,
            "notes": "Notebook faltante",
        }

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    first_cells = cells[:3]
    has_intro = any(
        cell.get("cell_type") == "markdown" and "".join(cell.get("source", [])).strip()
        for cell in first_cells
    )

    has_abs_paths = False
    error_outputs = 0
    large_outputs = 0

    for cell in cells:
        source_text = "".join(cell.get("source", []))
        if any(p.search(source_text) for p in ABSOLUTE_PATH_PATTERNS):
            has_abs_paths = True

        for output in cell.get("outputs", []):
            output_text = json.dumps(output, ensure_ascii=False)
            if any(p.search(output_text) for p in ABSOLUTE_PATH_PATTERNS):
                has_abs_paths = True
            if output.get("output_type") == "error":
                error_outputs += 1
            if len(output_text) > 200000:
                large_outputs += 1

    notes: list[str] = []
    if not has_intro:
        notes.append("Falta introduccion clara en primeras celdas")
    if has_abs_paths:
        notes.append("Contiene rutas absolutas en source u outputs")
    if error_outputs > 0:
        notes.append(f"Tiene {error_outputs} celdas con error")
    if large_outputs > 0:
        notes.append(f"Tiene {large_outputs} outputs grandes")

    return {
        "exists": True,
        "has_intro": has_intro,
        "has_abs_paths": has_abs_paths,
        "error_outputs": error_outputs,
        "large_outputs": large_outputs,
        "notes": "; ".join(notes) if notes else "OK",
    }


def expected_artifacts() -> list[ArtifactCheck]:
    return [
        ArtifactCheck("notebook", "01_data_understanding", "notebooks/01_data_understanding.ipynb", True),
        ArtifactCheck("notebook", "02_feature_engineering_modeling", "notebooks/02_feature_engineering_modeling.ipynb", True),
        ArtifactCheck("notebook", "02b_high_frequency_sensor_aggregation", "notebooks/02b_high_frequency_sensor_aggregation.ipynb", True),
        ArtifactCheck("notebook", "03_explainability", "notebooks/03_explainability.ipynb", True),
        ArtifactCheck("notebook", "04_simulation_what_if", "notebooks/04_simulation_what_if.ipynb", True),
        ArtifactCheck("notebook", "06_mlops_experiment_management", "notebooks/06_mlops_experiment_management.ipynb", True),
        ArtifactCheck("model", "selected_model_metadata", "models/selected/selected_model_metadata.json", True),
        ArtifactCheck("model", "feature_columns", "models/selected/feature_columns.json", True),
        ArtifactCheck("model", "feature_columns_strict_no_lab_input", "models/selected/feature_columns_strict_no_lab_input.json", True),
        ArtifactCheck("model", "model_lagged_lab_assumption", "models/selected/model_lagged_lab_assumption.pkl", True),
        ArtifactCheck("model", "model_strict_no_lab_input", "models/selected/model_strict_no_lab_input.pkl", True),
        ArtifactCheck("model", "lag_1_available_best_model", "models/selected/lag_1_available_best_model.pkl", True),
        ArtifactCheck("model", "lag_3_available_best_model", "models/selected/lag_3_available_best_model.pkl", True),
        ArtifactCheck("model", "lag_6_available_best_model", "models/selected/lag_6_available_best_model.pkl", True),
        ArtifactCheck("model", "no_recent_lab_available_best_model", "models/selected/no_recent_lab_available_best_model.pkl", True),
        ArtifactCheck("metric", "model_comparison", "reports/metrics/model_comparison.csv", True),
        ArtifactCheck("metric", "high_frequency_aggregation_comparison", "reports/metrics/high_frequency_aggregation_comparison.csv", True),
        ArtifactCheck("metric", "scenario_results", "reports/metrics/scenario_results.csv", True),
        ArtifactCheck("metric", "scenario_ranking", "reports/metrics/scenario_ranking.csv", True),
        ArtifactCheck("metric", "mlflow_runs_audit", "reports/metrics/mlflow_runs_audit.csv", True),
        ArtifactCheck("metric", "model_version_audit", "reports/metrics/model_version_audit.csv", True),
        ArtifactCheck("metric", "reproducibility_artifact_checklist", "reports/metrics/reproducibility_artifact_checklist.csv", True),
        ArtifactCheck("metric", "mlops_summary", "reports/metrics/mlops_summary.md", True),
        ArtifactCheck("figure", "raw_to_hourly_reduction", "reports/figures/raw_to_hourly_reduction.png", True),
        ArtifactCheck("figure", "target_distribution", "reports/figures/target_distribution.png", True),
        ArtifactCheck("figure", "target_temporal", "reports/figures/target_temporal.png", True),
        ArtifactCheck("figure", "correlation_matrix", "reports/figures/correlation_matrix.png", False, optional=True),
        ArtifactCheck("figure", "real_vs_predicted_test", "reports/figures/real_vs_predicted_test.png", True),
        ArtifactCheck("figure", "temporal_residuals", "reports/figures/temporal_residuals.png", True),
        ArtifactCheck("figure", "residuals_distribution", "reports/figures/residuals_distribution.png", True),
        ArtifactCheck("figure", "mae_by_month", "reports/figures/mae_by_month.png", False, optional=True),
        ArtifactCheck("figure", "high_frequency_aggregation_r2_comparison", "reports/figures/high_frequency_aggregation_r2_comparison.png", True),
        ArtifactCheck("figure", "high_frequency_aggregation_mae_comparison", "reports/figures/high_frequency_aggregation_mae_comparison.png", False, optional=True),
        ArtifactCheck("figure", "sensor_only_aggregated_feature_importance", "reports/figures/sensor_only_aggregated_feature_importance.png", True),
        ArtifactCheck("figure", "shap_bar", "reports/figures/shap_bar.png", True),
        ArtifactCheck("figure", "shap_summary", "reports/figures/shap_summary.png", True),
        ArtifactCheck("figure", "shap_waterfall_representative", "reports/figures/shap_waterfall_0_representative_case.png", True),
        ArtifactCheck("figure", "shap_waterfall_high_error", "reports/figures/shap_waterfall_0_high_error_case.png", True),
        ArtifactCheck("figure", "shap_waterfall_low_error", "reports/figures/shap_waterfall_0_low_error_case.png", False, optional=True),
        ArtifactCheck("figure", "pdp_silica_lag_1", "reports/figures/pdp_silica_concentrate_lag_1.png", True),
        ArtifactCheck("figure", "pdp_ore_pulp_ph", "reports/figures/pdp_ore_pulp_ph.png", True),
        ArtifactCheck("figure", "scenario_impact_heatmap", "reports/figures/scenario_impact_heatmap.png", True),
        ArtifactCheck("figure", "scenario_delta_bar_by_case", "reports/figures/scenario_delta_bar_by_case.png", False, optional=True),
        ArtifactCheck("figure", "tornado_sensitivity_median_case", "reports/figures/tornado_sensitivity_median_case.png", False, optional=True),
        ArtifactCheck("figure", "lab_delay_availability_r2", "reports/figures/lab_delay_availability_r2.png", True),
        ArtifactCheck("figure", "mlflow_ui_runs", "reports/figures/mlflow_ui_runs.png", False, optional=True),
        ArtifactCheck("presentation", "beamer_tex", "minsur_quality_prediction_beamer_v2.tex", False, optional=True),
        ArtifactCheck("presentation", "beamer_pdf", "minsur_quality_prediction_beamer_v2.pdf", False, optional=True),
        ArtifactCheck("api", "api_file", "src/api.py", False, optional=True),
        ArtifactCheck("api", "inference_file", "src/inference.py", False, optional=True),
        ArtifactCheck("api", "simulation_file", "src/simulation.py", False, optional=True),
        ArtifactCheck("demo", "streamlit_app", "app.py", False, optional=True),
    ]


def evaluate_status(exists: bool, critical: bool, optional: bool) -> str:
    if exists:
        return "complete"
    if optional:
        return "missing_optional"
    if critical:
        return "missing_critical"
    return "missing"


def api_endpoint_status(api_path: Path) -> dict[str, bool]:
    expected = {
        "/health": False,
        "/model-info": False,
        "/features": False,
        "/predict": False,
        "/simulate": False,
    }
    if not api_path.exists():
        return expected

    text = api_path.read_text(encoding="utf-8")
    for endpoint in expected:
        expected[endpoint] = endpoint in text
    return expected


def main() -> None:
    REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    notebook_paths = [
        PROJECT_ROOT / "notebooks" / "01_data_understanding.ipynb",
        PROJECT_ROOT / "notebooks" / "02_feature_engineering_modeling.ipynb",
        PROJECT_ROOT / "notebooks" / "02b_high_frequency_sensor_aggregation.ipynb",
        PROJECT_ROOT / "notebooks" / "03_explainability.ipynb",
        PROJECT_ROOT / "notebooks" / "04_simulation_what_if.ipynb",
        PROJECT_ROOT / "notebooks" / "06_mlops_experiment_management.ipynb",
    ]
    notebook_audit = {rel(p): notebook_quality_audit(p) for p in notebook_paths}

    rows: list[dict[str, str]] = []
    for item in expected_artifacts():
        exists = path_exists(item.expected_path)
        status = evaluate_status(exists, item.critical, item.optional)
        notes = item.notes

        if item.category == "notebook":
            qa = notebook_audit.get(item.expected_path, {})
            if qa:
                if qa.get("exists"):
                    extra = []
                    if not qa.get("has_intro", False):
                        extra.append("sin introduccion clara")
                    if qa.get("has_abs_paths", False):
                        extra.append("contiene rutas absolutas")
                    if qa.get("error_outputs", 0) > 0:
                        extra.append(f"errores_en_outputs={qa.get('error_outputs', 0)}")
                    if qa.get("large_outputs", 0) > 0:
                        extra.append(f"outputs_grandes={qa.get('large_outputs', 0)}")
                    notes = "; ".join(extra) if extra else "estructura base OK"
                    if extra and status == "complete":
                        status = "partial"
                else:
                    notes = "notebook faltante"

        rows.append(
            {
                "category": item.category,
                "artifact": item.artifact,
                "expected_path": item.expected_path,
                "exists": "yes" if exists else "no",
                "status": status,
                "notes": notes,
            }
        )

    api_checks = api_endpoint_status(PROJECT_ROOT / "src" / "api.py")
    for endpoint, ok in api_checks.items():
        rows.append(
            {
                "category": "api_endpoint",
                "artifact": endpoint,
                "expected_path": "src/api.py",
                "exists": "yes" if ok else "no",
                "status": "complete" if ok else "missing_optional",
                "notes": "endpoint detectado" if ok else "endpoint no detectado en api.py",
            }
        )

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "artifact", "expected_path", "exists", "status", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    complete = sum(1 for r in rows if r["status"] == "complete")
    partial = sum(1 for r in rows if r["status"] == "partial")
    missing_critical = sum(1 for r in rows if r["status"] == "missing_critical")
    missing_optional = sum(1 for r in rows if r["status"] == "missing_optional")

    presentation_required = [
        "reports/figures_presentation/raw_to_hourly_reduction.png",
        "reports/figures_presentation/high_frequency_aggregation_r2_comparison.png",
        "reports/figures_presentation/real_vs_predicted_test.png",
        "reports/figures_presentation/temporal_residuals.png",
        "reports/figures_presentation/shap_bar.png",
        "reports/figures_presentation/shap_waterfall_0_representative_case.png",
        "reports/figures_presentation/lab_delay_availability_r2.png",
        "reports/figures_presentation/scenario_impact_heatmap.png",
        "reports/figures_presentation/mlflow_ui_runs.png",
    ]
    presentation_found = [p for p in presentation_required if path_exists(p)]
    presentation_missing = [p for p in presentation_required if not path_exists(p)]

    claims_to_downgrade: list[str] = []
    if not path_exists("minsur_quality_prediction_beamer_v2.pdf"):
        claims_to_downgrade.append("No afirmar entrega de PDF final compilado en el repo.")
    if path_exists("src/api.py"):
        claims_to_downgrade.append("Presentar API como Nivel 7 opcional/demostrativo, no despliegue productivo.")

    notebooks_with_abs_paths = sum(
        1 for qa in notebook_audit.values() if qa.get("has_abs_paths", False)
    )

    with SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Resumen de auditoria del proyecto\n\n")
        f.write("## Estado general\n")
        f.write(f"- Completos: {complete}\n")
        f.write(f"- Parciales: {partial}\n")
        f.write(f"- Faltantes criticos: {missing_critical}\n")
        f.write(f"- Faltantes opcionales: {missing_optional}\n\n")

        f.write("## Hallazgos críticos\n")
        critical_rows = [r for r in rows if r["status"] == "missing_critical"]
        if critical_rows:
            for r in critical_rows:
                f.write(f"- {r['artifact']} -> {r['expected_path']} ({r['notes']})\n")
        else:
            f.write("- No se detectaron artefactos criticos faltantes en el inventario esperado.\n")
        f.write("\n")

        f.write("## Hallazgos parciales\n")
        partial_rows = [r for r in rows if r["status"] == "partial"]
        if partial_rows:
            for r in partial_rows:
                f.write(f"- {r['artifact']} -> {r['expected_path']} ({r['notes']})\n")
        else:
            f.write("- No se detectaron artefactos en estado parcial.\n")
        f.write("\n")

        f.write("## Auditoría de presentación (solo diagnóstico)\n")
        f.write(f"- Figuras disponibles: {len(presentation_found)}\n")
        f.write(f"- Figuras faltantes: {len(presentation_missing)}\n")
        if presentation_missing:
            for p in presentation_missing:
                f.write(f"  - Faltante: {p}\n")
        if claims_to_downgrade:
            f.write("- Afirmaciones que deben moderarse:\n")
            for c in claims_to_downgrade:
                f.write(f"  - {c}\n")
        f.write("\n")

        f.write("## Qué está completo\n")
        f.write("- Notebooks clave presentes.\n")
        f.write("- Artefactos de modelos seleccionados presentes.\n")
        f.write("- Figuras principales del flujo presentes.\n")
        f.write("- API y app demo presentes.\n\n")

        f.write("## Qué falta o requiere cuidado\n")
        if notebooks_with_abs_paths > 0:
            f.write("- Algunos notebooks incluyen rutas absolutas dentro de source u outputs serializados.\n")
        f.write("- La compilación PDF final no está como artefacto versionado en el repositorio.\n")
        f.write("- Revisar política de .gitignore para evidencias que deben versionarse.\n\n")

        f.write("## Comandos recomendados\n")
        f.write("```bash\n")
        f.write("python scripts/audit_project_outputs.py\n")
        f.write("```\n")

    print(f"Audit CSV generated: {REPORT_CSV}")
    print(f"Audit summary generated: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
