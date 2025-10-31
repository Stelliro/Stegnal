# sanitize_cupy_header.ps1  (PS 5.1-safe, idempotent)
param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py"
)

if (!(Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

# Backup
$bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
Copy-Item $FilePath $bak -Force
Write-Host "[backup] $bak"

# Load whole file
$raw = Get-Content $FilePath -Raw -Encoding UTF8

# 1) Normalize tabs -> spaces to avoid weird indents
$raw = $raw -replace "`t","    "

# 2) Define a clean, canonical stub
$cleanStub = @"
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

# 3) Remove any existing/broken stub block and stray 'import cupy' lines outside the stub
#    a) First, try to replace any existing stub region (from numpy-stub try to cp=None) with the clean one.
$pattern = '(?ms)^\s*try:\s*#\s*numpy-import-stub.*?#\s*cupy-import-stub.*?^\s*except\s+Exception:\s*\r?\n\s*cp\s*=\s*None\s*\r?\n'
$new = [regex]::Replace($raw, $pattern, $cleanStub, 'Multiline, Singleline')

if ($new -eq $raw) {
  # b) If no identifiable stub region, we will insert the stub after the initial import cluster.
  $lines = $raw -split "`r?`n"
  # strip ANY stray 'import cupy' lines (with or without alias), anywhere
  $filtered = New-Object System.Collections.Generic.List[string]
  foreach($ln in $lines){
    if($ln -match '^\s*import\s+cupy(\s+as\s+cp)?\s*$'){ continue }
    $filtered.Add($ln)
  }
  $lines = $filtered

  # find end of initial import cluster
  $lastImport = -1
  for($i=0; $i -lt $lines.Count; $i++){
    if($lines[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $lastImport = $i; continue }
    if($lastImport -ge 0 -and $lines[$i].Trim() -ne '' -and $lines[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
  }
  $ins = 0; if ($lastImport -ge 0) { $ins = $lastImport + 1 }

  # build injected content
  $block = @()
  if ($ins -gt 0 -and $lines[$ins-1].Trim() -ne '') { $block += '' }
  $block += ($cleanStub -split "`r?`n")
  $block += ''
  $lines = $lines[0..($ins-1)] + $block + $lines[$ins..($lines.Count-1)]
  $new = ($lines -join "`r`n")
} else {
  # c) After replacing the stub region, also strip *other* stray 'import cupy' lines that might still lurk.
  $new = ($new -split "`r?`n" | Where-Object {$_ -notmatch '^\s*import\s+cupy(\s+as\s+cp)?\s*$'} ) -join "`r`n"
}

# 4) Force CRLF and write back
$new = ($new -replace "`r?`n","`r`n")
Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
Write-Host "[patched] sanitized header + installed clean NumPy/CuPy stub"
