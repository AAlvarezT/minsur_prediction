
## Nivel 1 - Entendimiento de datos (obligatorio)
- Evidencia: encuadre temporal, variable objetivo, calidad y reduccion raw->hourly
- Ruta de artefactos: `notebooks/01_data_understanding.ipynb`, `reports/figures/raw_to_hourly_reduction.png`, `reports/figures/target_distribution.png`, `reports/figures/target_temporal.png`
- Estado: completo
- Comentario: evidencia disponible; auditar rutas absolutas en outputs serializados del notebook.

## Nivel 2 - Modelado temporal (obligatorio)
- Evidencia: split cronologico, lags, rolling features, benchmark y seleccion
- Ruta de artefactos: `notebooks/02_feature_engineering_modeling.ipynb`, `models/selected/selected_model_metadata.json`, `reports/figures/real_vs_predicted_test.png`, `reports/figures/temporal_residuals.png`
- Estado: completo
- Comentario: metrica y baseline preservados; revisar limpieza de outputs en notebook.

## Nivel 3 - Explicabilidad (obligatorio)
- Evidencia: SHAP global/local y PDP interpretado como comportamiento del modelo
- Ruta de artefactos: `notebooks/03_explainability.ipynb`, `reports/figures/shap_bar.png`, `reports/figures/shap_summary.png`, `reports/figures/shap_waterfall_0_representative_case.png`, `reports/figures/pdp_silica_concentrate_lag_1.png`
- Estado: completo
- Comentario: mantener disclaimer no causal en notebook/demo/presentacion.

## Nivel 6 - Reproducibilidad y MLOps (obligatorio)
- Evidencia: auditoria de runs/versiones, checklist y resumen MLOps
- Ruta de artefactos: `notebooks/06_mlops_experiment_management.ipynb`, `reports/metrics/mlflow_runs_audit.csv`, `reports/metrics/model_version_audit.csv`, `reports/metrics/reproducibility_artifact_checklist.csv`, `reports/metrics/mlops_summary.md`
- Estado: completo
- Comentario: trackeo local valido para evaluacion; no afirmar plataforma enterprise.

## Nivel 4 - Analisis de escenarios (opcional)
- Evidencia: escenarios what-if y ranking de sensibilidad
- Ruta de artefactos: `notebooks/04_simulation_what_if.ipynb`, `reports/metrics/scenario_results.csv`, `reports/metrics/scenario_ranking.csv`, `reports/figures/scenario_impact_heatmap.png`
- Estado: completo
- Comentario: presentar como sensibilidad predictiva, no recomendacion causal.

## Nivel 7 - Exposicion API (opcional)
- Evidencia: endpoints y contrato de inferencia/simulacion
- Ruta de artefactos: `src/api.py`, `src/inference.py`, `src/simulation.py`
- Estado: completo
- Comentario: mantener claim como demo/opcional; no afirmar despliegue productivo.

## Riesgo transversal: leakage operacional
- Evidencia: tabla de riesgo por grupo de features
- Ruta de artefactos: `reports/metrics/operational_leakage_audit.csv`
- Estado: completo
- Comentario: split temporal cubre leakage estadistico, pero inferencia exige auditoria de disponibilidad operacional.
