# add_gpu_error_class_precise.ps1
# PS 5.1-safe, surgical:
# - If class GPUAccelerationRequiredError is missing, insert a tiny definition
#   just after the import cluster and any simple top-level decls.
# - If __all__ exists, add the symbol there.
# - Otherwise, do nothing else. Includes -Verify mode.

param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py",
  [switch]$Verify
)

if (!(Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

# Read file
$raw   = Get-Content $FilePath -Raw -Encoding UTF8
$lines = $raw -split "`r?`n"

# Already present?
if ($raw -match '(?m)^\s*class\s+GPUAccelerationRequiredError\s*\(') {
  if ($Verify) { Write-Host "[verify] Class already present; no changes."; exit 0 }
  Write-Host "[no-op] Class already present; nothing to do."
  exit 0
}

# Find end of import cluster
$lastImport = -1
for($i=0; $i -lt $lines.Count; $i++){
  if ($lines[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)') { $lastImport = $i; continue }
  if ($lastImport -ge 0 -and $lines[$i].Trim() -ne '' -and $lines[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)') { break }
}

$ins = 0
if ($lastImport -ge 0) { $ins = $lastImport + 1 }

# Step past blank lines and simple module decls like NAME = None (keep it minimal)
while ($ins -lt $lines.Count) {
  $t = $lines[$ins].Trim()
  if ($t -eq '') { $ins++; continue }
  if ($t -match '^[A-Za-z_][A-Za-z0-9_]*\s*=\s*None\s*$') { $ins++; continue }
  break
}

# Prepare class block (4-space indent)
$cls = @"
class GPUAccelerationRequiredError(Exception):
    """Raised when GPU acceleration is required but unavailable."""
    pass

"@ -split "`r?`n"

if ($Verify) {
  Write-Host "[verify] Would insert GPUAccelerationRequiredError at ~line $($ins+1)"
  # Also check __all__
  if ($raw -match '(?m)^\s*__all__\s*=\s*\[' -and $raw -notmatch '(?m)\bGPUAccelerationRequiredError\b') {
    Write-Host "[verify] Would add 'GPUAccelerationRequiredError' to __all__ list."
  } else {
    Write-Host "[verify] __all__ update not needed (absent or already contains)."
  }
  exit 0
}

# Backup
$bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
Copy-Item $FilePath $bak -Force
Write-Host "[backup] $bak"

# Insert class
$lines = $lines[0..($ins-1)] + $cls + $lines[$ins..($lines.Count-1)]

# If __all__ exists, ensure symbol included
$start = -1; $end = -1
for($i=0; $i -lt $lines.Count; $i++){
  if ($lines[$i] -match '^\s*__all__\s*=\s*\[') { $start = $i; break }
}
if ($start -ge 0) {
  for($j=$start; $j -lt $lines.Count; $j++){
    if ($lines[$j] -match '\]') { $end = $j; break }
  }
  if ($end -ge 0) {
    $block = ($lines[$start..$end] -join "`n")
    if ($block -notmatch '\bGPUAccelerationRequiredError\b') {
      $indent = ($lines[$end] -replace '([^\s]).*$','')
      $lines = $lines[0..($end-1)] + @("$indent    'GPUAccelerationRequiredError',") + $lines[$end..($lines.Count-1)]
      Write-Host "[export] Added 'GPUAccelerationRequiredError' to __all__"
    }
  }
}

# Write back (CRLF)
($lines -join "`r`n") | Set-Content $FilePath -Encoding UTF8 -NoNewline
Write-Host "[patched] Inserted GPUAccelerationRequiredError cleanly."
