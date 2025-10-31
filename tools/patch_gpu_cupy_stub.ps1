# patch_gpu_cupy_stub.ps1
# PS 5.1 compatible, idempotent.
# Adds a CuPy import stub (cp = None if unavailable) and a small helper.

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

# 1) Ensure we already added the numpy stub earlier; if not, add it now
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

# 2) Insert CuPy stub if missing
$joined = ($lines -join "`n")
if ($joined -notmatch '(?m)^\s*#\s*cupy-import-stub' -and $joined -notmatch '(?m)^\s*import\s+cupy\s+as\s+cp') {
  $ins = Find-EndOfImports $lines
  # Place after numpy stub and any blank line(s)
  while ($ins -lt $lines.Count -and $lines[$ins].Trim() -eq '') { $ins++ }
  $blk = @(
    ""
    "# cupy-import-stub"
    "try:"
    "    import cupy as cp"
    "except Exception:"
    "    cp = None"
    ""
  )
  $lines = $lines[0..($ins-1)] + $blk + $lines[$ins..($lines.Count-1)]
  Write-Host "[insert] CuPy stub (cp)"
}

# 3) Insert get_array_module helper if missing
$joined = ($lines -join "`n")
if ($joined -notmatch '(?m)^\s*def\s+get_array_module\s*\(') {
  # Insert near other small helpers (after stubs we just added)
  # Find a reasonable insertion point: after import stubs and any simple decls
  $ins = Find-EndOfImports $lines
  while ($ins -lt $lines.Count) {
    $t = $lines[$ins].Trim()
    if ($t -eq '' -or $t -match '^\w[\w\d_]*\s*=\s*None\s*$' -or $t -match '^(try:|#\s*cupy-import-stub|try:\s*#\s*numpy-import-stub)') {
      $ins++
      continue
    }
    break
  }
  $blk = @(
    "def get_array_module(x=None):"
    "    """ + 'Return cp if available (and if x is a CuPy array), else NumPy.' + """"
    "    # If CuPy is present and x is a CuPy ndarray, prefer cp"
    "    if cp is not None:"
    "        try:"
    "            import cupy"
    "            if x is not None and isinstance(x, cupy.ndarray):"
    "                return cp"
    "            # Even if x is None, code that wants GPU can choose cp explicitly later"
    "            return cp"
    "        except Exception:"
    "            pass"
    "    # Fallback to numpy module alias _np"
    "    if _np is None:"
    "        raise RuntimeError(""numpy is required for CPU array ops"")"
    "    return _np"
    ""
  )
  $lines = $lines[0..($ins-1)] + $blk + $lines[$ins..($lines.Count-1)]
  Write-Host "[insert] get_array_module()"
}

# 4) Update __all__ if present
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
      if ($block -notmatch "(?m)\bcp\b") { $toAdd += "cp" }
      if ($block -notmatch "(?m)\bget_array_module\b") { $toAdd += "get_array_module" }
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
  Write-Host "[note] __all__ not found; export list unchanged (ok)."
}

# 5) Write back CRLF
$new = ($lines -join "`r`n")
if ($new -ne $raw) {
  Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
  Write-Host "[patched] $FilePath"
} else {
  Write-Host "[no-op] No changes required"
}
Write-Host "[done]"
