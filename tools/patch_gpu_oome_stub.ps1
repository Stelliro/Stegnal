# patch_gpu_oome_stub.ps1
# Adds is_cupy_out_of_memory_error() to gpu_runtime.py and exports it if __all__ exists.
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

# 1) Insert function if missing
if ($raw -notmatch '(?m)^\s*def\s+is_cupy_out_of_memory_error\s*\(') {
  # Insert after existing helper stubs (like get_array_module / allocate_pinned_array), or after imports.
  $insertIdx = Find-EndOfImports $lines
  # step past blank lines and simple decls
  while ($insertIdx -lt $lines.Count) {
    $t = $lines[$insertIdx].Trim()
    if ($t -eq '' -or $t -match '^\w[\w\d_]*\s*=\s*None\s*$' -or $t -match '^(try:|#\s*cupy-import-stub|try:\s*#\s*numpy-import-stub)') { $insertIdx++ } else { break }
  }
  # also step past small helpers if present
  $helperNames = @('get_array_module','allocate_pinned_array','free_pinned_array','is_gpu_available','require_gpu')
  $scan = $insertIdx
  while ($scan -lt $lines.Count) {
    $line = $lines[$scan]
    if ($line -match '^\s*def\s+(' + ($helperNames -join '|') + ')\s*\(') { $scan++ ; continue }
    break
  }
  $insertIdx = $scan

  $func = @(
    ""
    "def is_cupy_out_of_memory_error(exc):"
    "    """ + 'Return True if exc is a CuPy OOM error (any known variant); False otherwise.' + """"
    "    try:"
    "        import cupy"
    "        candidates = []"
    "        # Common CuPy OOM classes across versions:"
    "        try:"
    "            from cupy.cuda import memory as _cupy_mem"
    "            if hasattr(_cupy_mem, 'OutOfMemoryError'):"
    "                candidates.append(_cupy_mem.OutOfMemoryError)"
    "        except Exception:"
    "            pass"
    "        if hasattr(cupy, 'OutOfMemoryError'):"
    "            candidates.append(cupy.OutOfMemoryError)"
    "        try:"
    "            # Some builds expose via cupy.core.core"
    "            if hasattr(cupy, 'core') and hasattr(cupy.core, 'core') and hasattr(cupy.core.core, 'OutOfMemoryError'):"
    "                candidates.append(cupy.core.core.OutOfMemoryError)"
    "        except Exception:"
    "            pass"
    "        for cls in candidates:"
    "            try:"
    "                if isinstance(exc, cls):"
    "                    return True"
    "            except Exception:"
    "                pass"
    "        return False"
    "    except Exception:"
    "        # No CuPy or import failed => cannot be a CuPy OOM"
    "        return False"
    ""
  )

  $lines = $lines[0..($insertIdx-1)] + $func + $lines[$insertIdx..($lines.Count-1)]
  Write-Host "[insert] is_cupy_out_of_memory_error()"
}

# 2) Add to __all__ if present
$joined = ($lines -join "`n")
if ($joined -match '(?m)^\s*__all__\s*=\s*\[') {
  $lines = $joined -split "`r?`n"
  $start = -1; $end = -1
  for($i=0;$i -lt $lines.Count;$i++){ if($lines[$i] -match '^\s*__all__\s*=\s*\['){ $start = $i; break } }
  if ($start -ge 0) {
    for($j=$start; $j -lt $lines.Count; $j++){ if($lines[$j] -match '\]'){ $end = $j; break } }
    if ($end -ge 0) {
      $block = ($lines[$start..$end] -join "`n")
      if ($block -notmatch '\bis_cupy_out_of_memory_error\b') {
        $indent = ($lines[$end] -replace '([^\s]).*$','')
        $lines = $lines[0..($end-1)] + @("$indent    'is_cupy_out_of_memory_error',") + $lines[$end..($lines.Count-1)]
        Write-Host "[insert] __all__ += is_cupy_out_of_memory_error"
      }
    }
  }
} else {
  Write-Host "[note] __all__ not found; export list unchanged (ok)."
}

# 3) Write back with CRLF
$new = ($lines -join "`r`n")
if ($new -ne $raw) {
  Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
  Write-Host "[patched] $FilePath"
} else {
  Write-Host "[no-op] No changes required"
}
Write-Host "[done]"
