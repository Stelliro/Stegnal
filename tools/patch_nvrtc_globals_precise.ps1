# patch_nvrtc_globals_precise.ps1
# PS 5.1-safe, surgical patch.
# - Removes ONLY module-scope 'global' lines for specific NVRTC vars.
# - Inserts ONLY missing top-level declarations, right after import cluster.
# - Leaves everything else untouched.
param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py",
  [switch]$Verify  # If set, show intended changes without writing.
)

if (-not (Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

# --- Config ---
$nvNames = @(
  '_NVRTC_PATH_CACHED',
  '_NVRTC_DETECTED_LIBRARY',
  '_NVRTC_DETECTED_VERSION',
  '_NVRTC_VERSION_MATCHED'
)
$nvRegex = ($nvNames -join '|')

# --- Read (preserve CRLF) ---
$raw   = Get-Content $FilePath -Raw -Encoding UTF8
$lines = $raw -split "`r?`n"

# --- Step 1: remove ONLY module-level global lines that mention those names ---
$removeIdx = @()
for($i=0; $i -lt $lines.Count; $i++){
  $ln      = $lines[$i]
  $indent  = ($ln -replace '([^\s]).*$','').Length
  if($indent -eq 0 -and $ln -match "^\s*global\s+.*($nvRegex)"){
    $removeIdx += $i
  }
}

# --- Step 2: ensure top-level declarations (ONLY missing ones) ---
$needDecl = New-Object System.Collections.Generic.List[string]
foreach($n in $nvNames){
  if($raw -notmatch ("(?m)^\s*{0}\s*=" -f [regex]::Escape($n))){
    $needDecl.Add("$n = None")
  }
}

# Find end of contiguous import cluster (to place the decls)
function Get-EndOfImports([string[]]$L){
  $last = -1
  for($i=0; $i -lt $L.Count; $i++){
    if($L[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $last = $i; continue }
    if($last -ge 0 -and $L[$i].Trim() -ne '' -and $L[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
  }
  if($last -ge 0){ return $last + 1 } else { return 0 }
}

$insertIdx = $null
if($needDecl.Count -gt 0){
  $insertIdx = Get-EndOfImports $lines
}

# --- Preview / Apply ---
if($Verify){
  if($removeIdx.Count -gt 0){
    Write-Host "Would remove module-level global lines:" -ForegroundColor Yellow
    $removeIdx | ForEach-Object {
      Write-Host ("  line {0}: {1}" -f ($_+1), $lines[$_].Trim())
    }
  } else {
    Write-Host "No module-level global lines to remove." -ForegroundColor Green
  }

  if($needDecl.Count -gt 0){
    Write-Host ("Would insert {0} declaration(s) after imports at index {1} (1-based line ~{2}):" -f $needDecl.Count, $insertIdx, ($insertIdx+1)) -ForegroundColor Yellow
    $needDecl | ForEach-Object { Write-Host "  $_" }
  } else {
    Write-Host "All four NVRTC declarations already exist; none will be added." -ForegroundColor Green
  }
  exit 0
}

# Backup once if we’ll change anything
$willChange = ($removeIdx.Count -gt 0 -or $needDecl.Count -gt 0)
if($willChange){
  $bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
  Copy-Item $FilePath $bak -Force
  Write-Host "[backup] $bak"
}

# Apply removals (from bottom to top so indices stay valid)
if($removeIdx.Count -gt 0){
  foreach($idx in ($removeIdx | Sort-Object -Descending)){
    $lines = $lines[0..($idx-1)] + $lines[($idx+1)..($lines.Count-1)]
  }
  Write-Host "[remove] module-level global lines: $($removeIdx.Count)"
}

# Insert missing decls after imports
if($needDecl.Count -gt 0){
  $block = @()
  if($insertIdx -gt 0 -and $lines[$insertIdx-1].Trim() -ne ''){ $block += '' }
  $block += $needDecl
  $block += ''
  $lines = $lines[0..($insertIdx-1)] + $block + $lines[$insertIdx..($lines.Count-1)]
  Write-Host "[insert] decls: $($needDecl -join ', ')"
}

# Write back only if changed
if($willChange){
  ($lines -join "`r`n") | Set-Content $FilePath -Encoding UTF8 -NoNewline
  Write-Host "[done] gpu_runtime patched precisely."
} else {
  Write-Host "[no-op] Nothing to change."
}
