# Politica de carpeta de datos

Estructura:
- `raw/`: fuente original (no versionar archivos pesados)
- `interim/`: transformaciones intermedias (no versionar archivos pesados)
- `processed/`: datasets para modelado e inferencia (no versionar archivos pesados)

Regla:
- Mantener en git solo documentos livianos (`README.md`, `.gitkeep`) y evidencia auditada en `reports/`.
- No usar rutas absolutas al leer/escribir datos en notebooks o scripts.
- Resolver rutas desde `Path(__file__).resolve()` o `PROJECT_ROOT`.
