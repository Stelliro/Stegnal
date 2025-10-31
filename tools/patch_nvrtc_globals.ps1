# patch_nvrtc_globals.ps1
# PS 5.1-compatible. Idempotent.
# Fixes "name '_NVRTC_PATH_CACHED' is used prior to global declaration" in gpu_runtime.py

param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py"
)

function Get-Indent([string]$s){ ($s -replace '([^\s]).*$','').Length }

if (-not (Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = "$FilePath.$ts.bak"
Copy-Item $FilePath $backup -Force
Write-Host "[backup] -> $backup"

# Load file raw + lines
$raw   = Get-Content -Path $FilePath -Raw -Encoding UTF8
$lines = $raw -split "`r?`n"

# Targets
$nvVars = @('_NVRTC_PATH_CACHED','_NVRTC_DETECTED_LIBRARY','_NVRTC_DETECTED_VERSION','_NVRTC_VERSION_MATCHED')
$nvAlt  = ($nvVars -join '|')
$anyRefPattern = "\b($nvAlt)\b"

# ---------- Step 1: ensure module-level declarations ----------
$haveAll = $true
foreach($v in $nvVars){
  if(-not ($lines -match ("^\s*{0}\s*=" -f [regex]::Escape($v)))) { $haveAll = $false; break }
}

if(-not $haveAll){
  # find last import block index
  $lastImportIdx = -1
  for($i=0; $i -lt $lines.Count; $i++){
    if($lines[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $lastImportIdx = $i; continue }
    if($lastImportIdx -ge 0 -and $lines[$i].Trim() -ne '' -and $lines[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
  }

  $insertIdx = 0
  if($lastImportIdx -ge 0){ $insertIdx = $lastImportIdx + 1 }

  # build declarations that are missing
  $decls = @()
  foreach($v in $nvVars){
    if(-not ($lines -match ("^\s*{0}\s*=" -f [regex]::Escape($v)))){ $decls += "$v = None" }
  }
  if($decls.Count -gt 0){
    $blk = @()
    # spacer if needed
    if($insertIdx -gt 0 -and $lines[$insertIdx-1].Trim() -ne ''){ $blk += '' }
    $blk += $decls
    $blk += ''
    $lines = $lines[0..($insertIdx-1)] + $blk + $lines[$insertIdx..($lines.Count-1)]
    Write-Host "[insert] module-level: $($decls -join ', ')"
  }
}

# ---------- Step 2: add function-level 'global ...' where needed ----------
$idx = 0
while($idx -lt $lines.Count){
  $line = $lines[$idx]

  if($line -match '^\s*def\s+\w+\s*\(.*\)\s*:\s*$'){
    $funcStart = $idx
    $funcIndent = Get-Indent $line
    $bodyStart = $funcStart + 1

    # find end of function by indentation drop
    $funcEnd = $lines.Count - 1
    for($j=$bodyStart; $j -lt $lines.Count; $j++){
      $lt = $lines[$j].Trim()
      $li = Get-Indent $lines[$j]
      if($lt -eq '' -or $lt -like '#*'){ continue }
      if($li -le $funcIndent -and $j -gt $funcStart){ $funcEnd = $j - 1; break }
    }

    # scan for refs / existing global
    $hasRef = $false; $hasGlobal = $false
    for($k=$bodyStart; $k -le $funcEnd; $k++){
      $ln = $lines[$k]
      if($ln -match '^\s*global\s+' -and $ln -match $anyRefPattern){ $hasGlobal = $true }
      if($ln -match $anyRefPattern){ $hasRef = $true }
      if($hasRef -and $hasGlobal){ break }
    }

    if($hasRef -and -not $hasGlobal){
      $insertIndent = ' ' * ($funcIndent + 4)
      $globalLine = "$insertIndent" + 'global ' + ($nvVars -join ', ')
      # guard against double insert
      if($bodyStart -lt $lines.Count -and $lines[$bodyStart].Trim() -ne $globalLine.Trim()){
        $lines = $lines[0..$funcStart] + @($lines[$funcStart]) + @($globalLine) + $lines[$bodyStart..($lines.Count-1)]
        Write-Host ("[insert] global decl in function starting line {0}" -f ($funcStart+1))
        $idx += 1 # account for insertion
      }
    }

    $idx = $funcEnd + 1
    continue
  }
  $idx += 1
}

# ---------- Write back ----------
$new = ($lines -join "`r`n")
if($new -ne $raw){
  Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
  Write-Host "[patched] $FilePath"
}else{
  Write-Host "[no-op] Already compliant."
}
Write-Host "[done]"
