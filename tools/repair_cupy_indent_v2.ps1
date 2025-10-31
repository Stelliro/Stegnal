# repair_cupy_indent_v2.ps1
# PS 5.1-compatible. Rebuilds clean NumPy/CuPy import stubs and removes broken ones.

param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py"
)

if (-not (Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

# 0) Backup and normalize tabs
$bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
Copy-Item $FilePath $bak -Force
Write-Host "[backup] $bak"

$raw  = Get-Content $FilePath -Raw -Encoding UTF8
$norm = $raw -replace "`t","    "

# 1) Remove any previous/broken numpy/cupy stubs and stray 'import cupy as cp'
$lines = $norm -split "`r?`n"

# Helper to detect start/end of the initial import cluster
function Get-ImportEnd([string[]]$L){
  $last = -1
  for($i=0;$i -lt $L.Count;$i++){
    if($L[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $last = $i; continue }
    if($last -ge 0 -and $L[$i].Trim() -ne '' -and $L[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
  }
  if($last -ge 0){ return $last + 1 } else { return 0 }
}

# Strip any blocks that start with markers or a bare indented/dangling cupy import
$clean = New-Object System.Collections.Generic.List[string]
$skip = $false
$mode = ""
for($i=0;$i -lt $lines.Count;$i++){
  $ln = $lines[$i]

  if(-not $skip -and $ln -match '^\s*try:\s*#\s*numpy-import-stub'){ $skip=$true; $mode="numpy"; continue }
  if(-not $skip -and $ln -match '^\s*#\s*cupy-import-stub'){ $skip=$true; $mode="cupy"; continue }

  if($skip){
    # stop skipping at first blank line after block header
    if($ln.Trim() -eq ''){ $skip=$false; $mode=""; continue } else { continue }
  }

  # Remove stray/bare cupy import lines (indented or not)
  if($ln -match '^\s*import\s+cupy\s+as\s+cp\s*$'){ continue }

  $clean.Add($ln)
}

$lines = $clean

# 2) Recompute end of import cluster and rebuild clean stubs right after it
$ins = Get-ImportEnd $lines
$stub = @(
  ""
  "try:  # numpy-import-stub"
  "    import numpy as _np"
  "except Exception:"
  "    _np = None"
  ""
  "# cupy-import-stub"
  "try:"
  "    import cupy as cp"
  "except Exception:"
  "    cp = None"
  ""
)

if($ins -gt 0 -and $lines[$ins-1].Trim() -ne ''){ $stub = @('') + $stub }
$lines = $lines[0..($ins-1)] + $stub + $lines[$ins..($lines.Count-1)]

# 3) Write back with CRLF
$new = ($lines -join "`r`n")
if($new -ne $raw){
  Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
  Write-Host "[patched] rebuilt import stubs and removed stray cupy import"
}else{
  Write-Host "[no-op] no change"
}
Write-Host "[done]"
