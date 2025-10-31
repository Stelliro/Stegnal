# add_pinned_array_stubs_precise.ps1
# PS 5.1-safe, surgical, idempotent.
# - Adds allocate_pinned_array / free_pinned_array if missing.
# - Ensures a minimal NumPy alias (_np) exists (non-fatal if import fails).
# - Updates __all__ if present. Leaves everything else untouched.

param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py",
  [switch]$Verify
)

if (!(Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

$raw   = Get-Content $FilePath -Raw -Encoding UTF8
$lines = $raw -split "`r?`n"

# Helpers
function Find-EndOfImports([string[]]$L){
  $last = -1
  for($i=0; $i -lt $L.Count; $i++){
    if($L[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $last = $i; continue }
    if($last -ge 0 -and $L[$i].Trim() -ne '' -and $L[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
  }
  if($last -ge 0){ return $last + 1 } else { return 0 }
}

$hasAlloc = ($raw -match '(?m)^\s*def\s+allocate_pinned_array\s*\(')
$hasFree  = ($raw -match '(?m)^\s*def\s+free_pinned_array\s*\(')
$hasNP    = ($raw -match '(?m)^\s*import\s+numpy\s+as\s+_np') -or ($raw -match '(?m)^\s*try:\s*#\s*numpy-import-stub')

$toInsert = @()

# 1) Ensure a minimal NumPy alias (_np) exists
if (-not $hasNP) {
  $npStub = @(
    "",
    "try:  # numpy-import-stub",
    "    import numpy as _np",
    "except Exception:",
    "    _np = None",
    ""
  )
  # insert after import cluster
  $ins = Find-EndOfImports $lines
  if ($ins -gt 0 -and $lines[$ins-1].Trim() -ne '') { $npStub = @('') + $npStub }
  $lines = $lines[0..($ins-1)] + $npStub + $lines[$ins..($lines.Count-1)]
}

# 2) Build stubs as needed
$stub = @()
if (-not $hasAlloc) {
  $stub += @(
    "def allocate_pinned_array(shape, dtype=""float32""):",
    "    """"CPU fallback: return a normal NumPy array as a stand-in for pinned memory.""""",
    "    if _np is None:",
    "        raise RuntimeError(""numpy is required for allocate_pinned_array"")",
    "    return _np.zeros(shape, dtype=dtype)",
    ""
  )
}
if (-not $hasFree) {
  $stub += @(
    "def free_pinned_array(arr):",
    "    """"CPU fallback: nothing to free; rely on GC.""""",
    "    return None",
    ""
  )
}

# 3) Insert stubs in a sensible place:
#    after GPUAccelerationRequiredError class if present, else after imports & simple decls.
if ($stub.Count -gt 0) {
  $insertIdx = -1
  for($i=0; $i -lt $lines.Count; $i++){
    if($lines[$i] -match '^\s*class\s+GPUAccelerationRequiredError\b'){
      $insertIdx = $i + 1
      # skip through the class body to next blank line
      for($j=$insertIdx; $j -lt $lines.Count; $j++){
        if($lines[$j].Trim() -eq ''){ $insertIdx = $j + 1; break }
      }
      break
    }
  }
  if ($insertIdx -lt 0) {
    $insertIdx = Find-EndOfImports $lines
    # step past blank lines and NAME=None decls
    while ($insertIdx -lt $lines.Count) {
      $t = $lines[$insertIdx].Trim()
      if ($t -eq '' -or $t -match '^[A-Za-z_][A-Za-z0-9_]*\s*=\s*None\s*$') { $insertIdx++ } else { break }
    }
  }
  $lines = $lines[0..($insertIdx-1)] + $stub + $lines[$insertIdx..($lines.Count-1)]
}

# 4) If __all__ exists, add symbols (only if missing)
$joined = ($lines -join "`n")
if ($joined -match '(?m)^\s*__all__\s*=\s*\[') {
  $lines = $joined -split "`r?`n"
  $start = -1; $end = -1
  for($i=0; $i -lt $lines.Count; $i++){ if($lines[$i] -match '^\s*__all__\s*=\s*\['){ $start = $i; break } }
  if ($start -ge 0) {
    for($j=$start; $j -lt $lines.Count; $j++){ if($lines[$j] -match '\]'){ $end = $j; break } }
    if ($end -ge 0) {
      $block = ($lines[$start..$end] -join "`n")
      $toAdd = @()
      if ($block -notmatch '\ballocate_pinned_array\b' -and -not $hasAlloc){ $toAdd += 'allocate_pinned_array' }
      if ($block -notmatch '\bfree_pinned_array\b'     -and -not $hasFree ){ $toAdd += 'free_pinned_array' }
      if ($toAdd.Count -gt 0) {
        $indent = ($lines[$end] -replace '([^\s]).*$','')
        foreach($name in $toAdd){
          $lines = $lines[0..($end-1)] + @("$indent    '$name',") + $lines[$end..($lines.Count-1)]
          $end += 1
        }
      }
    }
  }
}

if ($Verify) {
  Write-Host "[verify] allocate_pinned_array present? $hasAlloc"
  Write-Host "[verify] free_pinned_array present? $hasFree"
  Write-Host "[verify] numpy alias present? $hasNP"
  exit 0
}

# Write back only if changed
$new = ($lines -join "`r`n")
if ($new -ne $raw) {
  $bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
  Copy-Item $FilePath $bak -Force
  Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
  Write-Host "[patched] Added missing pinned-array stubs (backup: $bak)"
} else {
  Write-Host "[no-op] Nothing to change."
}
