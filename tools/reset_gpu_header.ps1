# reset_gpu_header.ps1
# PS 5.1 compatible. Rebuilds clean NumPy/CuPy stubs and removes/neutralizes any dangling 'try:' in the header.

param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py"
)

if (-not (Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

# 0) Backup + normalize tabs
$bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
Copy-Item $FilePath $bak -Force
Write-Host "[backup] $bak"

$raw  = Get-Content $FilePath -Raw -Encoding UTF8
$norm = $raw -replace "`t","    "
$lines = $norm -split "`r?`n"

function Get-ImportEnd([string[]]$L){
  $last = -1
  for($i=0;$i -lt $L.Count;$i++){
    if($L[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $last = $i; continue }
    if($last -ge 0 -and $L[$i].Trim() -ne '' -and $L[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
  }
  if($last -ge 0){ return $last + 1 } else { return 0 }
}

# 1) Remove previous/broken numpy/cupy stub blocks and stray 'import cupy as cp'
$clean = New-Object System.Collections.Generic.List[string]
$skipping = $false
$mode = ""
for($i=0;$i -lt $lines.Count;$i++){
  $ln = $lines[$i]

  if(-not $skipping -and $ln -match '^\s*try:\s*#\s*numpy-import-stub'){ $skipping=$true; $mode="numpy";  continue }
  if(-not $skipping -and $ln -match '^\s*#\s*cupy-import-stub'){        $skipping=$true; $mode="cupy";   continue }

  if($skipping){
    # Skip until first blank separator after the block
    if($ln.Trim() -eq ''){ $skipping=$false; $mode=""; continue } else { continue }
  }

  # Remove stray/bare cupy import lines anywhere
  if($ln -match '^\s*import\s+cupy\s+as\s+cp\s*$'){ continue }

  $clean.Add($ln)
}
$lines = $clean

# 2) Guard against a dangling 'try:' in the first 80 lines (header)
$lastTryIdx = -1
for($i=0; $i -lt [Math]::Min(80, $lines.Count); $i++){
  $t = $lines[$i].Trim()
  if($t -like 'try:*'){ $lastTryIdx = $i; continue }
  if($t -like 'except*' -or $t -like 'finally*'){ $lastTryIdx = -1; continue }
  if(($t -like 'def *' -or $t -like 'class *') -and $lastTryIdx -ge 0){
    # Neutralize the dangling try
    $lines[$lastTryIdx] = "# FIX: removed dangling try: " + $lines[$lastTryIdx].Trim()
    $lastTryIdx = -1
    break
  }
}

# 3) Rebuild clean stubs right after the import cluster
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

# 4) Write back (CRLF)
$new = ($lines -join "`r`n")
if($new -ne $raw){
  Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
  Write-Host "[patched] rebuilt header stubs and removed stray/dangling blocks"
}else{
  Write-Host "[no-op] no change"
}
Write-Host "[done]"
