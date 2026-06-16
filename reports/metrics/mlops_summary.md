# Resumen de auditoria MLOps (Nivel 6)

## Modelo seleccionado
- Modelo seleccionado: Extra Trees Regressor
- Rol final: recommended_model_with_lagged_lab_assumption

## Modelo fallback
- Modelo fallback: Random Forest
- Rol final: strict_no_lab_input_fallback

## Resumen de metricas
- Mejor RMSE de validacion (modelo seleccionado): 0.6629
- RMSE final en test (modelo seleccionado): 0.7496
- R2 final en test (modelo seleccionado): 0.6083

## Supuesto central
- El modelo seleccionado asume disponibilidad de laboratorio rezagado al momento de inferencia.

## Artefactos generados por esta auditoria
- mlflow_runs_audit.csv
- mlflow_leaderboard_audit.csv
- model_version_audit.csv
- reproducibility_artifact_checklist.csv
- selected_model_metadata_audit.csv
- feature_consistency_audit.csv
- operational_assumptions_audit.csv
- mlops_limitations_audit.csv
- production_readiness_roadmap.csv
- level6_checklist_audit.csv

## Estado de reproducibilidad
- Estado del checklist de artefactos: PARCIAL

## Limitaciones
- MLflow es local (sin gobierno centralizado).
- Model Registry con stages/aliases no esta implementado en este prototipo.
- Aun no hay reentrenamiento automatico ni monitoreo online de drift.
- Las salidas de escenarios son predictivas, no recomendaciones causales.

## Siguientes pasos
- Centralizar backend de MLflow.
- Implementar Model Registry formal.
- Validar SLA de delay de laboratorio con operaciones.
- Construir flujos de inferencia monitorizada y reentrenamiento.
