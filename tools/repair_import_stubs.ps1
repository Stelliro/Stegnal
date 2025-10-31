# repair_import_stubs.ps1
# PS 5.1 compatible. Idempotent.
# Rewrites the numpy/cupy import stubs in gpu_runtime.py so they are valid.

param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py"
)

if (-not (Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

# Backup
$bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
Copy-Item $FilePath $bak -Force
Write-Host "[backup] $bak"

# Load & normalize tabs->spaces
$raw = Get-Content $FilePath -Raw -Encoding UTF8
$norm = $raw -replace "`t", "    "

# Helper: remove a block starting at a marker line (inclusive) through the next blank line
function Remove-BlockByMarker([string]$text, [string]$markerRegex){
  $lines = $text -split "`r?`n"
  $out = New-Object System.Collections.Generic.List[string]
  $skip = $false
  $skipping = $false
  for($i=0; $i -lt $lines.Count; $i++){
    if(-not $skipping -and $lines[$i] -match $markerRegex){
      $skipping = $true
      continue
    }
    if($skipping){
      # stop skipping after first blank line following the marker block
      if($lines[$i].Trim() -eq ''){
        $skipping = $false
        continue
      } else {
        continue
      }
    }
    $out.Add($lines[$i])
  }
  return ($out -join "`r`n")
}

# Remove any previous/broken stubs
$norm = Remove-BlockByMarker $norm '^\s*try:\s*#\s*numpy-import-stub'
$norm = Remove-BlockByMarker $norm '^\s*#\s*cupy-import-stub'

# Find end of import cluster
$lines = $norm -split "`r?`n"
$lastImport = -1
for($i=0; $i -lt $lines.Count; $i++){
  if($lines[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $lastImport = $i; continue }
  if($lastImport -ge 0 -and $lines[$i].Trim() -ne '' -and $lines[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
}
if ($lastImport -ge 0) { $ins = $lastImport + 1 } else { $ins = 0 }

# Build clean stubs
$stubBlock = @(
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

# Insert stubs right after imports, keeping spacing
if($ins -gt 0 -and $lines[$ins-1].Trim() -ne ''){ $stubBlock = @('') + $stubBlock }
$lines = $lines[0..($ins-1)] + $stubBlock + $lines[$ins..($lines.Count-1)]

# Write back with CRLF
$new = ($lines -join "`r`n")
if($new -ne $raw){
  Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
  Write-Host "[patched] rebuilt numpy/cupy stubs"
}else{
  Write-Host "[no-op] no changes"
}
Write-Host "[done]"
