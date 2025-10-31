# patch_nvrtc_globals_min.ps1  (PS 5.1-safe, idempotent)
param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py"
)

if (-not (Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

# 1) Backup
$bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
Copy-Item $FilePath $bak -Force
Write-Host "[backup] $bak"

# 2) Remove any module-scope 'global ...' lines that touch NVRTC names
$names = '_NVRTC_PATH_CACHED|_NVRTC_DETECTED_LIBRARY|_NVRTC_DETECTED_VERSION|_NVRTC_VERSION_MATCHED'
$lines = Get-Content $FilePath -Encoding UTF8
$kept  = @()
for($i=0; $i -lt $lines.Count; $i++){
  $ln = $lines[$i]
  $indent = ($ln -replace '([^\s]).*$','').Length
  if($indent -eq 0 -and $ln -match "^\s*global\s+.*($names)"){
    Write-Host ("[remove] line {0}: {1}" -f ($i+1), $ln.Trim())
    continue
  }
  $kept += $ln
}
$kept | Set-Content $FilePath -Encoding UTF8 -NoNewline
Write-Host "[pass] removed module-scope global if present"

# 3) Ensure top-level declarations exist once (insert after import cluster if missing)
$txt = Get-Content $FilePath -Raw -Encoding UTF8
$need = @(
  "_NVRTC_PATH_CACHED = None",
  "_NVRTC_DETECTED_LIBRARY = None",
  "_NVRTC_DETECTED_VERSION = None",
  "_NVRTC_VERSION_MATCHED = None"
) | Where-Object {
  $name = ($_ -split '\s*=')[0]
  $txt -notmatch ("(?m)^\s*{0}\s*=" -f [regex]::Escape($name))
}

if($need.Count -gt 0){
  $lines = $txt -split "`r?`n"
  # find end of import cluster
  $lastImport = -1
  for($i=0; $i -lt $lines.Count; $i++){
    if($lines[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $lastImport = $i; continue }
    if($lastImport -ge 0 -and $lines[$i].Trim() -ne '' -and $lines[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
  }
  if ($lastImport -ge 0) { $ins = $lastImport + 1 } else { $ins = 0 }

  $blk = @()
  if($ins -gt 0 -and $lines[$ins-1].Trim() -ne ''){ $blk += '' }
  $blk += $need
  $blk += ''
  $lines = $lines[0..($ins-1)] + $blk + $lines[$ins..($lines.Count-1)]
  ($lines -join "`r`n") | Set-Content $FilePath -Encoding UTF8 -NoNewline
  Write-Host "[insert] module-level decls: $($need -join ', ')"
}

Write-Host "[done] patch_nvrtc_globals_min.ps1"
