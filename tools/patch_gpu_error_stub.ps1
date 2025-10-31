# patch_gpu_error_stub.ps1
# PS 5.1-compatible. Idempotent.
# Adds a minimal GPUAccelerationRequiredError to gpu_runtime.py if missing
# and appends it to __all__ if that list exists.

$fp = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py"
if (-not (Test-Path $fp)) { Write-Error "File not found: $fp"; exit 1 }

# Backup
$backup = "$fp.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
Copy-Item $fp $backup -Force
Write-Host "[backup] $backup"

# Load raw text and lines
$raw   = Get-Content $fp -Raw -Encoding UTF8
$lines = $raw -split "`r?`n"

# 1) If class already exists, do nothing; else insert a minimal definition after imports / decls.
if ($raw -notmatch '(?m)^\s*class\s+GPUAccelerationRequiredError\s*\(') {
  # Find end of initial import cluster and top-level constant/decl block
  $lastImport = -1
  for ($i=0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)') { $lastImport = $i; continue }
    if ($lastImport -ge 0 -and $lines[$i].Trim() -ne '' -and $lines[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)') { break }
  }
  # Try to place the class after imports and any immediate blank spacer
  if ($lastImport -ge 0) { $ins = $lastImport + 1 } else { $ins = 0 }

  # Step slightly past any blank spacer and simple module-level = None decls
  while ($ins -lt $lines.Count) {
    $t = $lines[$ins].Trim()
    if ($t -eq '') { $ins++; continue }
    if ($t -match '^\w[\w\d_]*\s*=\s*None\s*$') { $ins++; continue }
    break
  }

  $classBlock = @(
    ""
    "class GPUAccelerationRequiredError(Exception):"
    "    """"Raised when GPU acceleration is required but unavailable."""""
    "    pass"
    ""
  )
  $lines = $lines[0..($ins-1)] + $classBlock + $lines[$ins..($lines.Count-1)]
  Write-Host "[insert] GPUAccelerationRequiredError class at ~line $($ins+1)"
} else {
  Write-Host "[skip] GPUAccelerationRequiredError already defined"
}

# 2) If __all__ exists, ensure the symbol is exported
$allIdx = -1
for ($i=0; $i -lt $lines.Count; $i++) {
  if ($lines[$i] -match '^\s*__all__\s*=\s*\[') { $allIdx = $i; break }
}
if ($allIdx -ge 0) {
  # gather lines until closing bracket
  $start = $allIdx
  $end = $start
  while ($end -lt $lines.Count) {
    if ($lines[$end] -match '\]') { break }
    $end++
  }
  $block = $lines[$start..$end]
  $joined = ($block -join "`n")
  if ($joined -notmatch "\bGPUAccelerationRequiredError\b") {
    # insert before the closing ']'
    for ($j=$end; $j -ge $start; $j--) {
      if ($lines[$j] -match '\]') {
        $indent = ($lines[$j] -replace '([^\s]).*$','')
        $lines = $lines[0..($j-1)] + @("$indent    'GPUAccelerationRequiredError',") + $lines[$j..($lines.Count-1)]
        Write-Host "[insert] Added 'GPUAccelerationRequiredError' to __all__"
        break
      }
    }
  } else {
    Write-Host "[skip] __all__ already includes GPUAccelerationRequiredError"
  }
} else {
  Write-Host "[note] __all__ not found; export list not modified (fine)."
}

# Write back if changed
$new = ($lines -join "`r`n")
if ($new -ne $raw) {
  Set-Content -Path $fp -Value $new -Encoding UTF8 -NoNewline
  Write-Host "[patched] $fp"
} else {
  Write-Host "[no-op] No changes needed"
}
Write-Host "[done]"
