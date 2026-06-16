# Repositorio auditable del soft sensor de flotacion de Minsur

Proyecto de prueba tecnica para predecir `% Silica Concentrate` en planta de flotacion usando un enfoque basado en datos, temporal y auditable.

## Objetivo del caso
Construir un sistema de alerta temprana (soft sensor) para `% Silica Concentrate` que sea:
- tecnicamente defendible,
- trazable en sus supuestos,
- reproducible por evaluadores externos,
- y separable en componentes (modelado, explicabilidad, escenarios y MLOps).

## Resultado principal (sin cambios)
Modelo recomendado original: Random Forest bajo supuesto de laboratorio rezagado disponible.

- Test MAE: ~0.5500
- Test RMSE: ~0.7474
- Test R2: ~0.6096

Comparativos clave:
- RF con sensores agregados + laboratorio rezagado: MAE ~0.5409, RMSE ~0.7399, R2 ~0.6190
- Baseline de persistencia: RMSE ~0.7614, R2 ~0.5965
- Modelo solo sensores agregado: MAE ~0.8921, RMSE ~1.1197, R2 ~0.1260

Sensibilidad por retraso de laboratorio (validacion):
- lag_1_available: R2 ~0.5876
- lag_3_available: R2 ~0.2626
- lag_6_available: R2 ~0.0486
- no_recent_lab_available: R2 ~0.0595

## Narrativa tecnica valida
Este sistema debe presentarse como soft sensor / alerta temprana para `% Silica Concentrate`.

No debe presentarse como:
- control automatico,
- recomendacion causal,
- ni despliegue productivo final.

## Estructura del proyecto
```text
minsur_prediction/
  data/
    README.md
    raw/
    interim/
    processed/
  notebooks/
    01_data_understanding.ipynb
    02_feature_engineering_modeling.ipynb
    02b_high_frequency_sensor_aggregation.ipynb
    03_explainability.ipynb
    04_simulation_what_if.ipynb
    06_mlops_experiment_management.ipynb
  src/
    __init__.py
    config.py
    data_preprocessing.py
    feature_engineering.py
    inference.py
    simulation.py
    api.py
    app_utils.py
    evaluate.py
    explain.py
    predict.py
    train.py
  models/
    selected/
  reports/
    metrics/
    figures/
    figures_presentation/
  scripts/
    collect_presentation_figures.py
    audit_project_outputs.py
  app.py
  minsur_quality_prediction_beamer_v2.tex
  requirements.txt
  .gitignore
```

## Orden de reproduccion recomendado
1. `notebooks/01_data_understanding.ipynb`
2. `notebooks/02_feature_engineering_modeling.ipynb`
3. `notebooks/02b_high_frequency_sensor_aggregation.ipynb`
4. `notebooks/03_explainability.ipynb`
5. `notebooks/04_simulation_what_if.ipynb`
6. `notebooks/06_mlops_experiment_management.ipynb`

## Salidas esperadas por notebook
### 01_data_understanding
- `reports/figures/raw_to_hourly_reduction.png`
- `reports/figures/target_distribution.png`
- `reports/figures/target_temporal.png`
- `reports/figures/correlation_matrix.png`

### 02_feature_engineering_modeling
- `models/selected/selected_model_metadata.json`
- `models/selected/feature_columns.json`
- `models/selected/feature_columns_strict_no_lab_input.json`
- `models/selected/*.pkl`
- `reports/figures/real_vs_predicted_test.png`
- `reports/figures/temporal_residuals.png`
- `reports/figures/residuals_distribution.png`
- `reports/figures/mae_by_month.png`

### 02b_high_frequency_sensor_aggregation
- `reports/metrics/high_frequency_aggregation_comparison.csv`
- `reports/figures/high_frequency_aggregation_r2_comparison.png`
- `reports/figures/high_frequency_aggregation_mae_comparison.png`
- `reports/figures/sensor_only_aggregated_feature_importance.png`

### 03_explainability
- `reports/figures/shap_bar.png`
- `reports/figures/shap_summary.png`
- `reports/figures/shap_waterfall_0_representative_case.png`
- `reports/figures/shap_waterfall_0_high_error_case.png`
- `reports/figures/shap_waterfall_0_low_error_case.png`
- `reports/figures/pdp_silica_concentrate_lag_1.png`
- `reports/figures/pdp_ore_pulp_ph.png`

### 04_simulation_what_if
- `reports/metrics/scenario_results.csv`
- `reports/metrics/scenario_ranking.csv`
- `reports/figures/scenario_impact_heatmap.png`
- `reports/figures/scenario_delta_bar_by_case.png`
- `reports/figures/tornado_sensitivity_median_case.png`

### 06_mlops_experiment_management
- `reports/metrics/mlflow_runs_audit.csv`
- `reports/metrics/model_version_audit.csv`
- `reports/metrics/reproducibility_artifact_checklist.csv`
- `reports/metrics/mlops_summary.md`
- `reports/figures/mlflow_ui_runs.png` (si existe captura)

## Auditoria del proyecto
Ejecutar:
```bash
python scripts/audit_project_outputs.py
```

Genera:
- `reports/metrics/project_audit_report.csv`
- `reports/metrics/project_audit_summary.md`

## Supuestos operativos
- El modelo recomendado usa laboratorio rezagado.
- `lag_1` equivale aproximadamente a una hora en la tabla horaria.
- Es valido solo si el laboratorio rezagado esta disponible antes de inferencia.
- Sin laboratorio reciente se debe usar fallback (menor desempeno).
- Los escenarios what-if se interpretan como sensibilidad predictiva, no causalidad.
- No es control automatico.

## Nivel 7 (API) - opcional
Si se evalua API:
```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Rutas de API esperadas:
- `/health`
- `/model-info`
- `/features`
- `/predict`
- `/simulate`

## Demo de Streamlit (opcional)
```bash
streamlit run app.py
```

La app consume `src/inference.py` y `src/simulation.py` y no debe duplicar logica de entrenamiento.

## MLflow local
```bash
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5001
```

## Requisitos
Instalacion minima:
```bash
pip install -r requirements.txt
```

## Limitaciones y afirmaciones que NO se deben hacer
- No afirmar causalidad fisica del proceso con SHAP/PDP.
- No afirmar despliegue productivo final.
- No afirmar recomendacion operativa automatica.
- No afirmar robustez fuera del rango historico de entrenamiento.
