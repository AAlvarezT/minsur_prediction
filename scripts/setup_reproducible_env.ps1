param(
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

function Resolve-PythonCommand {
    param([string]$RawPythonCmd)

    if (-not [string]::IsNullOrWhiteSpace($RawPythonCmd)) {
        $parts = $RawPythonCmd.Trim().Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
        if ($parts.Count -eq 0) {
            throw "PythonCmd invalido"
        }
        return @{ Exe = $parts[0]; Args = if ($parts.Count -gt 1) { $parts[1..($parts.Count - 1)] } else { @() } }
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ Exe = "py"; Args = @("-3.11") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Exe = "python"; Args = @() }
    }
    if (Get-Command python3 -ErrorAction SilentlyContinue) {
        return @{ Exe = "python3"; Args = @() }
    }

    throw "No se encontro un ejecutable de Python (py/python/python3)"
}

$pythonCmdResolved = Resolve-PythonCommand -RawPythonCmd $PythonCmd
$pythonExe = $pythonCmdResolved.Exe
$pythonArgs = $pythonCmdResolved.Args

Write-Host "[1/4] Creando entorno virtual $VenvDir..."
if (-not (Test-Path $VenvDir)) {
    Invoke-Checked $pythonExe @pythonArgs "-m" "venv" $VenvDir
}

$venvPython = "$VenvDir/Scripts/python.exe"
if (-not (Test-Path $venvPython)) {
    throw "No se encontro $venvPython"
}

Write-Host "[2/4] Actualizando pip/setuptools/wheel..."
Invoke-Checked $venvPython "-m" "pip" "install" "--upgrade" "pip" "setuptools" "wheel"

Write-Host "[3/4] Instalando dependencias (lockfile preferido)..."
if ($Perfil -eq "evaluador") {
    if (Test-Path "requirements-evaluator.txt") {
        Invoke-Checked $venvPython "-m" "pip" "install" "-r" "requirements-evaluator.txt"
    } else {
        throw "No existe requirements-evaluator.txt"
    }
} else {
    if (Test-Path "requirements-lock.txt") {
        Invoke-Checked $venvPython "-m" "pip" "install" "-r" "requirements-lock.txt"
    } else {
        Invoke-Checked $venvPython "-m" "pip" "install" "-r" "requirements.txt"
    }
}

Write-Host "[4/4] Validando instalacion minima..."
if ($Perfil -eq "evaluador") {
    Invoke-Checked $venvPython "-c" "import pandas, matplotlib; print('OK: entorno evaluador listo')"
} else {
    Invoke-Checked $venvPython "-c" "import numpy, pandas, sklearn, shap, mlflow, fastapi; print('OK: entorno completo listo')"
}

Write-Host "Listo. Usa: $venvPython scripts/audit_project_outputs.py"
