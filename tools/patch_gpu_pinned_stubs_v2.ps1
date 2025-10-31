# patch_gpu_pinned_stubs_v2.ps1
# PS 5.1 compatible, idempotent
# Adds CPU fallbacks for allocate_pinned_array/free_pinned_array and simple GPU shims.
# Also ensures a numpy import stub and updates __all__ if present.

param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py"
)

if (-not (Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

# Backup
$bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
Copy-Item $FilePath $bak -Force
Write-Host "[backup] $bak"

# Load
$raw   = Get-Content $FilePath -Raw -Encoding UTF8
$lines = $raw -split "`r?`n"

function Find-EndOfImports([string[]]$L){
  $last = -1
  for($i=0; $i -lt $L.Count; $i++){
    if($L[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $last = $i; continue }
    if($last -ge 0 -and $L[$i].Trim() -ne '' -and $L[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
  }
  if($last -ge 0){ return $last + 1 } else { return 0 }
}

# ---------- 1) Ensure numpy import stub ----------
if ($raw -notmatch '(?m)^\s*import\s+numpy\s+as\s+_np') {
  $ins = Find-EndOfImports $lines
  $blk = @(
    ""
    "try:  # numpy-import-stub"
    "    import numpy as _np"
    "except Exception:"
    "    _np = None"
    ""
  )
  $lines = $lines[0..($ins-1)] + $blk + $lines[$ins..($lines.Count-1)]
  Write-Host "[insert] numpy import stub"
}

# ---------- 2) Prepare stub functions ----------
$needAlloc   = ($raw -notmatch '(?m)^\s*def\s+allocate_pinned_array\s*\(')
$needFree    = ($raw -notmatch '(?m)^\s*def\s+free_pinned_array\s*\(')
$needAvail   = ($raw -notmatch '(?m)^\s*def\s+is_gpu_available\s*\(')
$needRequire = ($raw -notmatch '(?m)^\s*def\s+require_gpu\s*\(')

$stub = @()

if ($needAlloc) {
  $stub += @(
    "def allocate_pinned_array(shape, dtype=""float32""):"
    "    """ + 'CPU fallback: return a normal NumPy array as a stand-in for pinned memory.' + """"
    "    if _np is None:"
    "        raise RuntimeError(""numpy is required for allocate_pinned_array"")"
    "    return _np.zeros(shape, dtype=dtype)"
    ""
  )
}
if ($needFree) {
  $stub += @(
    "def free_pinned_array(arr):"
    "    """ + 'CPU fallback: nothing to free; rely on GC.' + """"
    "    return None"
    ""
  )
}
if ($needAvail) {
  $stub += @(
    "def is_gpu_available():"
    "    """ + 'Conservative default: GPU unavailable unless runtime sets this true.' + """"
    "    return False"
    ""
  )
}
if ($needRequire) {
  $stub += @(
    "def require_gpu():"
    "    """ + 'Raise standardized error when GPU is required but not available.' + """"
    "    raise GPUAccelerationRequiredError(""GPU acceleration required but unavailable"")"
    ""
  )
}

if ($stub.Count -gt 0) {
  # Insert after our GPUAccelerationRequiredError class if present,
  # else after imports and any simple "= None" decls.
  $insertIdx = -1
  for($i=0;$i -lt $lines.Count;$i++){
    if($lines[$i] -match '^\s*class\s+GPUAccelerationRequiredError\b'){
      $insertIdx = $i + 1
      for($j=$insertIdx; $j -lt $lines.Count; $j++){
        if($lines[$j].Trim() -eq '') { $insertIdx = $j + 1; break }
      }
      break
    }
  }
  if ($insertIdx -lt 0) {
    $insertIdx = Find-EndOfImports $lines
    while ($insertIdx -lt $lines.Count) {
      $t = $lines[$insertIdx].Trim()
      if ($t -eq '' -or $t -match '^\w[\w\d_]*\s*=\s*None\s*$') { $insertIdx++ } else { break }
    }
  }
  $lines = $lines[0..($insertIdx-1)] + $stub + $lines[$insertIdx..($lines.Count-1)]
  Write-Host "[insert] stubs: $((@($needAlloc,'alloc'),@($needFree,'free'),@($needAvail,'avail'),@($needRequire,'require') | % {$_}) -join ' ')"
}

# ---------- 3) Ensure __all__ includes symbols if __all__ exists ----------
$joined = ($lines -join "`n")
if ($joined -match '(?m)^\s*__all__\s*=\s*\[') {
  $lines = $joined -split "`r?`n"
  $start = -1; $end = -1
  for($i=0;$i -lt $lines.Count;$i++){ if($lines[$i] -match '^\s*__all__\s*=\s*\['){ $start = $i; break } }
  if ($start -ge 0) {
    for($j=$start; $j -lt $lines.Count; $j++){ if($lines[$j] -match '\]'){ $end = $j; break } }
    if ($end -ge 0) {
      $block = ($lines[$start..$end] -join "`n")
      $toAdd = @()
      if ($block -notmatch '\ballocate_pinned_array\b'){ $toAdd += 'allocate_pinned_array' }
      if ($block -notmatch '\bfree_pinned_array\b'){ $toAdd += 'free_pinned_array' }
      if ($block -notmatch '\bis_gpu_available\b'){ $toAdd += 'is_gpu_available' }
      if ($block -notmatch '\brequire_gpu\b'){ $toAdd += 'require_gpu' }
      if ($toAdd.Count -gt 0) {
        $indent = ($lines[$end] -replace '([^\s]).*$','')
        foreach($name in $toAdd){
          $lines = $lines[0..($end-1)] + @("$indent    '$name',") + $lines[$end..($lines.Count-1)]
          $end += 1
        }
        Write-Host "[insert] __all__ += $($toAdd -join ', ')"
      }
    }
  }
} else {
  Write-Host "[note] __all__ not found; export list unchanged."
}

# ---------- 4) Write back with CRLF ----------
$new = ($lines -join "`r`n")
if ($new -ne $raw) {
  Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
  Write-Host "[patched] $FilePath"
} else {
  Write-Host "[no-op] No changes required"
}
Write-Host "[done]"
