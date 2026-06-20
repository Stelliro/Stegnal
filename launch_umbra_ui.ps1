<#
  launch_umbra_ui.ps1 — PowerShell launcher for the Umbra desktop explorer.

  Mirrors launch_umbra_ui.bat: prefers a dedicated .venv (Python 3.12), creating
  and populating it on first run, then launches `python -m umbra ui`. Validates
  the venv by actually running it, so a venv whose base Python was removed is
  rebuilt instead of failing later.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvDir = Join-Path $PSScriptRoot ".venv"
$venvPy  = Join-Path $venvDir "Scripts\python.exe"

function Test-PyOk($exe) {
    if (-not (Test-Path $exe)) { return $false }
    try { & $exe --version *> $null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}

if (-not (Test-PyOk $venvPy)) {
    if (Test-Path $venvDir) {
        Write-Host "[WARN] Existing .venv is broken (its base Python was removed). Rebuilding..."
        Remove-Item -Recurse -Force $venvDir
    }

    $base = $null
    $local312 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path $local312) { $base = $local312 }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { $base = "py" }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $base = (Get-Command python).Source }

    if (-not $base) { Write-Error "No suitable Python found. Please install Python 3.12."; exit 1 }

    Write-Host "[SYSTEM] Creating virtual environment (.venv)..."
    if ($base -eq "py") { & py -3.12 -m venv $venvDir } else { & $base -m venv $venvDir }
    & $venvPy -m pip install --upgrade pip *> $null
}

# Ensure the package is importable; install on first run.
& $venvPy -c "import umbra" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[SYSTEM] Installing Umbra + UI dependencies (first run; may take a minute)..."
    & $venvPy -m pip install -e ".[ui]"
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to install dependencies."; exit $LASTEXITCODE }
}

& $venvPy -m umbra ui @Args
