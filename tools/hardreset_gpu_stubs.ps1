# hardreset_gpu_stubs.ps1
# PS 5.1-compatible. Replaces any broken numpy/cupy stub region with a valid block
# and removes stray 'import cupy as cp' lines that aren't inside the stub.

param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py"
)

if (!(Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

# Backup
$bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
Copy-Item $FilePath $bak -Force
Write-Host "[backup] $bak"

# Load raw text
$raw = Get-Content $FilePath -Raw -Encoding UTF8

# Regex: grab from the numpy-stub 'try:' through the cp = None line (inclusive).
# This nukes any malformed middle and re-writes it cleanly.
$pattern = '(?ms)^\s*try:\s*#\s*numpy-import-stub.*?#\s*cupy-import-stub.*?^\s*except\s+Exception:\s*\r?\n\s*cp\s*=\s*None\s*\r?\n'
$replacement = @"
try:  # numpy-import-stub
    import numpy as _np
except Exception:
    _np = None

# cupy-import-stub
try:
    import cupy as cp
except Exception:
    cp = None

"@

$new = [regex]::Replace($raw, $pattern, $replacement, 'Multiline, Singleline')

# If the pattern didn't match (e.g., stubs scattered), forcibly insert the block after top imports
if ($new -eq $raw) {
  $lines = $raw -split "`r?`n"

  # 1) Strip any stray/bare 'import cupy as cp' line (outside try/except)
  $clean = New-Object System.Collections.Generic.List[string]
  foreach ($ln in $lines) {
    if ($ln -match '^\s*import\s+cupy\s+as\s+cp\s*$') { continue }
    $clean.Add($ln)
  }
  $lines = $clean

  # 2) Find end of initial import cluster
  $lastImport = -1
  for($i=0; $i -lt $lines.Count; $i++){
    if($lines[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $lastImport = $i; continue }
    if($lastImport -ge 0 -and $lines[$i].Trim() -ne '' -and $lines[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
  }
  $ins = 0; if ($lastImport -ge 0) { $ins = $lastImport + 1 }

  # 3) Insert a blank line + clean stub block
  $block = @()
  if ($ins -gt 0 -and $lines[$ins-1].Trim() -ne '') { $block += '' }
  $block += $replacement.TrimEnd("`r","`n").Split("`n")
  $block += ''

  $lines = $lines[0..($ins-1)] + $block + $lines[$ins..($lines.Count-1)]
  $new = ($lines -join "`r`n")
  Write-Host "[insert] injected clean numpy/cupy stub after imports"
} else {
  Write-Host "[replace] rebuilt numpy/cupy stub block in-place"
}

# Write back with CRLF
Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
Write-Host "[patched] $FilePath"
