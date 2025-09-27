param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

Set-Location -LiteralPath $PSScriptRoot

# Resolve Python command (prefer "python", fallback to "py")
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $python) {
    Write-Error "Could not find Python on PATH. Please install Python 3.9+ and try again."
    exit 1
}
$pythonPath = $python.Source

# Ensure required dependencies are available
try {
    & $pythonPath -m pip install --upgrade pip | Out-Null
} catch {
    Write-Warning "Failed to upgrade pip. Continuing with existing version."
}

& $pythonPath -m pip install --upgrade --quiet --editable "$PSScriptRoot" 2>&1 | ForEach-Object { $_ | Write-Output }
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install Project Umbra dependencies."
    exit $LASTEXITCODE
}

# Launch the Project Umbra UI via the CLI wrapper
& $pythonPath -m umbra ui @Args
