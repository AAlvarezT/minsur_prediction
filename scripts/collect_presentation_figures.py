from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class FigureSpec:
    figure_name: str
    expected_slide: str
    expected_source_notebook: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_PRESENTATION_DIR = REPORTS_DIR / "figures_presentation"
AUDIT_CSV = METRICS_DIR / "presentation_figures_audit.csv"

SEARCH_DIRS = [
    REPORTS_DIR / "figures",
    PROJECT_ROOT / "figures",
    PROJECT_ROOT,
    PROJECT_ROOT / "notebooks" / "reports" / "figures",
    PROJECT_ROOT / "reports" / "figures",
]


def expected_specs() -> list[FigureSpec]:
    return [
        FigureSpec("raw_to_hourly_reduction.png", "Level 1 framing", "01_data_understanding.ipynb"),
        FigureSpec("target_distribution.png", "Appendix C", "01_data_understanding.ipynb"),
        FigureSpec("target_temporal.png", "Appendix C", "01_data_understanding.ipynb"),
        FigureSpec("correlation_matrix.png", "Appendix optional", "01_data_understanding.ipynb"),
        FigureSpec("real_vs_predicted_test.png", "Performance operacional", "02_feature_engineering_modeling.ipynb"),
        FigureSpec("real_vs_predicted_val.png", "Validation appendix", "02_feature_engineering_modeling.ipynb"),
        FigureSpec("temporal_residuals.png", "Performance operacional", "02_feature_engineering_modeling.ipynb"),
        FigureSpec("residuals_distribution.png", "Appendix D", "02_feature_engineering_modeling.ipynb"),
        FigureSpec("mae_by_month.png", "Appendix D", "02_feature_engineering_modeling.ipynb"),
        FigureSpec("main_vs_fallback_simulation.png", "Appendix optional", "02_feature_engineering_modeling.ipynb"),
        FigureSpec("high_frequency_aggregation_r2_comparison.png", "Benchmark comparison", "02b_high_frequency_sensor_aggregation.ipynb"),
        FigureSpec("high_frequency_aggregation_mae_comparison.png", "Appendix optional", "02b_high_frequency_sensor_aggregation.ipynb"),
        FigureSpec("sensor_only_aggregated_feature_importance.png", "Appendix B", "02b_high_frequency_sensor_aggregation.ipynb"),
        FigureSpec("shap_bar.png", "Explainability global", "03_explainability.ipynb"),
        FigureSpec("shap_summary.png", "Appendix E", "03_explainability.ipynb"),
        FigureSpec("shap_waterfall_0_representative_case.png", "Explainability local", "03_explainability.ipynb"),
        FigureSpec("shap_waterfall_0_high_error_case.png", "Appendix E", "03_explainability.ipynb"),
        FigureSpec("shap_waterfall_0_low_error_case.png", "Appendix E", "03_explainability.ipynb"),
        FigureSpec("pdp_silica_concentrate_lag_1.png", "Appendix F", "03_explainability.ipynb"),
        FigureSpec("pdp_ore_pulp_ph.png", "Appendix F", "03_explainability.ipynb"),
        FigureSpec("pdp_ore_pulp_ph_lag_1.png", "Appendix optional", "03_explainability.ipynb"),
        FigureSpec("pdp_silica_concentrate_lag_3.png", "Appendix optional", "03_explainability.ipynb"),
        FigureSpec("lab_delay_availability_r2.png", "Lab delay sensitivity", "04_simulation_what_if.ipynb"),
        FigureSpec("scenario_impact_heatmap.png", "Scenario analysis", "04_simulation_what_if.ipynb"),
        FigureSpec("scenario_delta_bar_by_case.png", "Appendix optional", "04_simulation_what_if.ipynb"),
        FigureSpec("tornado_sensitivity_median_case.png", "Scenario fallback", "04_simulation_what_if.ipynb"),
        FigureSpec("mlflow_ui_runs.png", "MLOps evidence", "06_mlops_experiment_management.ipynb"),
    ]


def unique_dirs(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        normalized = str(path.resolve()) if path.exists() else str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def find_figure(figure_name: str, search_dirs: list[Path]) -> Path | None:
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue

        direct = base_dir / figure_name
        if direct.exists() and direct.is_file():
            return direct

        matches = list(base_dir.rglob(figure_name))
        if matches:
            return matches[0]
    return None


def create_missing_figure_notice(path: Path, figure_name: str) -> None:
    plt.figure(figsize=(8, 2), dpi=150)
    plt.axis("off")
    plt.text(
        0.5,
        0.5,
        f"Figura no disponible: falta artefacto fuente\n{figure_name}",
        ha="center",
        va="center",
        fontsize=10,
        color="#6B7280",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def build_mlflow_audit_summary(output_path: Path) -> bool:
    runs_csv = METRICS_DIR / "mlflow_runs_audit.csv"
    model_csv = METRICS_DIR / "model_version_audit.csv"

    if not runs_csv.exists() and not model_csv.exists():
        return False

    runs_df = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame()
    model_df = pd.read_csv(model_csv) if model_csv.exists() else pd.DataFrame()

    summary_lines: list[str] = []
    if not runs_df.empty:
        summary_lines.append(f"Runs audit rows: {len(runs_df)}")
        cols = [c for c in ["run_id", "model_name", "mae", "rmse", "r2"] if c in runs_df.columns]
        if cols:
            sample = runs_df[cols].head(6).fillna("-")
            summary_lines.append(sample.to_string(index=False))
    if not model_df.empty:
        summary_lines.append("")
        summary_lines.append(f"Model versions rows: {len(model_df)}")
        cols = [c for c in ["model_name", "model_version", "stage", "status"] if c in model_df.columns]
        if cols:
            sample = model_df[cols].head(6).fillna("-")
            summary_lines.append(sample.to_string(index=False))

    if not summary_lines:
        return False

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    ax.axis("off")
    ax.set_title("MLflow Audit Summary", fontsize=16, fontweight="bold", loc="left")
    ax.text(
        0.01,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def collect_figures() -> None:
    FIGURES_PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    specs = expected_specs()
    search_dirs = unique_dirs(SEARCH_DIRS)

    # Build MLflow fallback only when UI screenshot is missing.
    mlflow_ui = find_figure("mlflow_ui_runs.png", search_dirs)
    mlflow_fallback_generated = False
    if mlflow_ui is None:
        fallback_path = FIGURES_PRESENTATION_DIR / "mlflow_audit_summary.png"
        mlflow_fallback_generated = build_mlflow_audit_summary(fallback_path)

    rows: list[dict[str, str]] = []
    for spec in specs:
        source_path = find_figure(spec.figure_name, search_dirs)
        destination = FIGURES_PRESENTATION_DIR / spec.figure_name

        found = source_path is not None
        action_required = ""

        if found and source_path is not None:
            shutil.copy2(source_path, destination)
        else:
            create_missing_figure_notice(destination, spec.figure_name)
            action_required = "Regenerar desde notebook fuente o artefacto métrico"

        rows.append(
            {
                "figure_name": spec.figure_name,
                "expected_slide": spec.expected_slide,
                "expected_source_notebook": spec.expected_source_notebook,
                "found": "yes" if found else "no",
                "source_path": str(source_path.relative_to(PROJECT_ROOT)) if found and source_path is not None else "",
                "copied_to": str(destination.relative_to(PROJECT_ROOT)),
                "action_required": action_required,
            }
        )

    if mlflow_ui is None:
        destination = FIGURES_PRESENTATION_DIR / "mlflow_audit_summary.png"
        if mlflow_fallback_generated and destination.exists():
            rows.append(
                {
                    "figure_name": "mlflow_audit_summary.png",
                    "expected_slide": "MLOps evidence fallback",
                    "expected_source_notebook": "06_mlops_experiment_management.ipynb or reports/metrics",
                    "found": "generated",
                    "source_path": "reports/metrics/mlflow_runs_audit.csv | reports/metrics/model_version_audit.csv",
                    "copied_to": str(destination.relative_to(PROJECT_ROOT)),
                    "action_required": "",
                }
            )
        else:
            create_missing_figure_notice(destination, "mlflow_audit_summary.png")
            rows.append(
                {
                    "figure_name": "mlflow_audit_summary.png",
                    "expected_slide": "MLOps evidence fallback",
                    "expected_source_notebook": "06_mlops_experiment_management.ipynb or reports/metrics",
                    "found": "no",
                    "source_path": "",
                    "copied_to": str(destination.relative_to(PROJECT_ROOT)),
                    "action_required": "Falta mlflow_ui_runs.png y no hay CSV de auditoría suficiente",
                }
            )

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "figure_name",
                "expected_slide",
                "expected_source_notebook",
                "found",
                "source_path",
                "copied_to",
                "action_required",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len([r for r in rows if r["figure_name"].endswith(".png") and r["figure_name"] != "mlflow_audit_summary.png"]) 
    found_count = sum(1 for r in rows if r["found"] in {"yes", "generated"})
    missing_count = sum(1 for r in rows if r["found"] == "no")
    print(f"Collected figure audit generated: {AUDIT_CSV}")
    print(f"Total tracked entries: {len(rows)}")
    print(f"Found or generated: {found_count}")
    print(f"Missing source artifacts: {missing_count}")


if __name__ == "__main__":
    collect_figures()
