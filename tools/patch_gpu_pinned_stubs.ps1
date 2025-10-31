# patch_gpu_pinned_stubs.ps1
# PS 5.1 compatible. Idempotent.
# Adds allocate_pinned_array / free_pinned_array (CPU fallback) and exports them.

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

# ---------- Ensure numpy import (lazy try/except block) ----------
if ($raw -notmatch '(?m)^\s*try:\s*#\s*numpy-import-stub') {
  # find end of import cluster
  $lastImport = -1
  for($i=0; $i -lt $lines.Count; $i++){
    if($lines[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $lastImport = $i; continue }
    if($lastImport -ge 0 -and $lines[$i].Trim() -ne '' -and $lines[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
  }
  if ($lastImport -ge 0) { $ins = $lastImport + 1 } else { $ins = 0 }

  $blk = @(
    ""
    "try:  # numpy-import-stub"
    "    import numpy as _np"
    "except Exception:  # keep import failure non-fatal; callers may not need it"
    "    _np = None"
    ""
  )
  $lines = $lines[0..($ins-1)] + $blk + $lines[$ins..($lines.Count-1)]
  Write-Host "[insert] numpy import stub"
}

# ---------- Insert allocate/free pinned stubs if missing ----------
$needAlloc = ($raw -notmatch '(?m)^\s*def\s+allocate_pinned_array\s*\(')
$needFree  = ($raw -notmatch '(?m)^\s*def\s+free_pinned_array\s*\(')

if ($needAlloc -or $needFree) {
  # Insert after our GPUAccelerationRequiredError class if present, else after imports+decls
  $insertIdx = -1
  for($i=0;$i -lt $lines.Count;$i++){
    if($lines[$i] -match '^\s*class\s+GPUAccelerationRequiredError\b'){
      # move insert pos to after the class block
      $insertIdx = $i + 1
      # skip its body
      for($j=$insertIdx; $j -lt $lines.Count; $j++){
        if($lines[$j].Trim() -eq '') { $insertIdx = $j + 1; break }
      }
      break
    }
  }
  if ($insertIdx -lt 0) {
    # fallback: after imports and simple None decls
    $lastImport = -1
    for($i=0;$i -lt $lines.Count;$i++){
      if($lines[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $lastImport = $i; continue }
      if($lastImport -ge 0 -and $lines[$i].Trim() -ne '' -and $lines[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
    }
    if ($lastImport -ge 0) { $insertIdx = $lastImport + 1 } else { $insertIdx = 0 }
    while ($insertIdx -lt $lines.Count) {
      $t = $lines[$insertIdx].Trim()
      if ($t -eq '' -or $t -match '^\w[\w\d_]*\s*=\s*None\s*$') { $insertIdx++ } else { break }
    }
  }

  $stub = @()
  if ($needAlloc) {
    $stub += @(
      "def allocate_pinned_array(shape, dtype=""float32""):"
      "    """ + 'Return a CPU array; acts as a pinned-mem stand-in when CUDA is unavailable.' + """"
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
      "    # No-op on CPU fallback."
      "    return None"
      ""
    )
  }

  $lines = $lines[0..($insertIdx-1)] + $stub + $lines[$insertIdx..($lines.Count-1)]
  Write-Host ("[insert] {0}{1}" -f ($needAlloc ? "allocate_pinned_array " : ""), ($needFree ? "free_pinned_array" : ""))
}

# ---------- Add to __all__ if present ----------
$joined = ($lines -join "`n")
if ($joined -match '(?m)^\s*__all__\s*=\s*\[') {
  # re-split by lines for editing
  $lines = $joined -split "`r?`n"
  # locate __all__ block
  $start = -1; $end = -1
  for($i=0;$i -lt $lines.Count;$i++){
    if($lines[$i] -match '^\s*__all__\s*=\s*\['){ $start = $i; break }
  }
  if ($start -ge 0) {
    for($j=$start; $j -lt $lines.Count; $j++){
      if($lines[$j] -match '\]'){ $end = $j; break }
    }
    if ($end -ge 0) {
      $block = $lines[$start..$end] -join "`n"
      $toAdd = @()
      if ($block -notmatch '\ballocate_pinned_array\b'){ $toAdd += "allocate_pinned_array" }
      if ($block -notmatch '\bfree_pinned_array\b'){ $toAdd += "free_pinned_array" }
      if ($toAdd.Count -gt 0) {
        # insert just before closing bracket
        $indent = ($lines[$end] -replace '([^\s]).*$','')
        foreach($name in $toAdd){
          $lines = $lines[0..($end-1)] + @("$indent    '$name',") + $lines[$end..($lines.Count-1)]
          $end += 1
        }
        Write-Host "[insert] added to __all__: $($toAdd -join ', ')"
      }
    }
  }
} else {
  Write-Host "[note] __all__ not found; export list unchanged (ok)."
}

# ---------- Write back (CRLF) ----------
$new = ($lines -join "`r`n")
if ($new -ne $raw) {
  Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
  Write-Host "[patched] $FilePath"
} else {
  Write-Host "[no-op] No changes required"
}
