param(
    [switch]$SkipSetup,
    [ValidateSet("evaluador", "completo")]
    [string]$Perfil = "evaluador",
    [string]$PythonCmd = "",
    [string]$VenvDir = ".venv_eval"
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Exe,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo comando: $Exe $($Args -join ' ')"
    }
}

if (-not $SkipSetup) {
    Write-Host "Paso 1: setup reproducible de entorno"
    ./scripts/setup_reproducible_env.ps1 -Perfil $Perfil -PythonCmd $PythonCmd -VenvDir $VenvDir
}

$venvPython = "$VenvDir/Scripts/python.exe"
if (-not (Test-Path $venvPython)) {
    throw "No existe $VenvDir. Ejecuta primero scripts/setup_reproducible_env.ps1"
}

Write-Host "Paso 2: auditoria de artefactos"
Invoke-Checked $venvPython "scripts/audit_project_outputs.py"

Write-Host "Paso 3: recopilacion de figuras para presentacion"
Invoke-Checked $venvPython "scripts/collect_presentation_figures.py"

if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
    Write-Host "Paso 4: compilacion opcional de Beamer"
    Invoke-Checked "pdflatex" "-interaction=nonstopmode" "-halt-on-error" "minsur_quality_prediction_beamer_v2.tex"
    Write-Host "PDF generado: minsur_quality_prediction_beamer_v2.pdf"
} else {
    Write-Host "Paso 4 omitido: pdflatex no disponible"
}

Write-Host "Listo: revisar reports/metrics/project_audit_summary.md"
