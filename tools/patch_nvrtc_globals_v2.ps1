# patch_nvrtc_globals_v2.ps1
# PS 5.1 compatible. Idempotent.
# Fixes: module-level 'global ...' and ensures proper function-level 'global' placement (after docstring).

param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py"
)

function Get-Indent([string]$s){ ($s -replace '([^\s]).*$','').Length }

if (-not (Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = "$FilePath.$ts.bak"
Copy-Item $FilePath $backup -Force
Write-Host "[backup] -> $backup"

$raw   = Get-Content -Path $FilePath -Raw -Encoding UTF8
$lines = $raw -split "`r?`n"

$nvVars = @('_NVRTC_PATH_CACHED','_NVRTC_DETECTED_LIBRARY','_NVRTC_DETECTED_VERSION','_NVRTC_VERSION_MATCHED')
$nvAlt  = ($nvVars -join '|')
$refPat = "\b($nvAlt)\b"
$asgPat = "^\s*($nvAlt)\s*="

# 1) Ensure module-level declarations exist once
$haveAll = $true
foreach($v in $nvVars){
  if(-not ($lines -match ("^\s*{0}\s*=" -f [regex]::Escape($v)))) { $haveAll = $false; break }
}

if(-not $haveAll){
  # find end of imports
  $lastImportIdx = -1
  for($i=0; $i -lt $lines.Count; $i++){
    if($lines[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $lastImportIdx = $i; continue }
    if($lastImportIdx -ge 0 -and $lines[$i].Trim() -ne '' -and $lines[$i] -notmatch '^\s*(import\s+|from\s+\S+\s+import\s+)'){ break }
  }
  $insertIdx = 0; if($lastImportIdx -ge 0){ $insertIdx = $lastImportIdx + 1 }

  $decls = @()
  foreach($v in $nvVars){
    if(-not ($lines -match ("^\s*{0}\s*=" -f [regex]::Escape($v)))){ $decls += "$v = None" }
  }
  if($decls.Count -gt 0){
    $blk = @()
    if($insertIdx -gt 0 -and $lines[$insertIdx-1].Trim() -ne ''){ $blk += '' }
    $blk += $decls
    $blk += ''
    $lines = $lines[0..($insertIdx-1)] + $blk + $lines[$insertIdx..($lines.Count-1)]
    Write-Host "[insert] module-level decls: $($decls -join ', ')"
  }
}

# 2) Remove any module-level 'global ...' lines (indent == 0)
$toRemove = @()
for($i=0; $i -lt $lines.Count; $i++){
  if((Get-Indent $lines[$i]) -eq 0 -and $lines[$i] -match '^\s*global\s+'){
    if($lines[$i] -match $refPat){
      $toRemove += $i
    }
  }
}
if($toRemove.Count -gt 0){
  foreach($idx in ($toRemove | Sort-Object -Descending)){
    $lines = $lines[0..($idx-1)] + $lines[($idx+1)..($lines.Count-1)]
  }
  Write-Host "[remove] module-level global lines: $($toRemove.Count)"
}

# 3) For each function that ASSIGNS to any _NVRTC_* var, ensure a global line after docstring
$idx = 0
while($idx -lt $lines.Count){
  $line = $lines[$idx]
  if($line -match '^\s*def\s+\w+\s*\(.*\)\s*:\s*$'){
    $funcStart = $idx
    $funcIndent = Get-Indent $line
    $bodyStart = $funcStart + 1

    # find function end by indentation drop
    $funcEnd = $lines.Count - 1
    for($j=$bodyStart; $j -lt $lines.Count; $j++){
      $lt = $lines[$j].Trim()
      $li = Get-Indent $lines[$j]
      if($lt -eq '' -or $lt -like '#*'){ continue }
      if($li -le $funcIndent -and $j -gt $funcStart){ $funcEnd = $j - 1; break }
    }

    # scan body for assignment to target names
    $assigns = $false
    for($k=$bodyStart; $k -le $funcEnd; $k++){
      if($lines[$k] -match $asgPat) { $assigns = $true; break }
    }

    if($assigns){
      # detect existing global in this function
      $hasGlobal = $false
      for($k=$bodyStart; $k -le $funcEnd; $k++){
        if($lines[$k] -match '^\s*global\s+' -and $lines[$k] -match $refPat){ $hasGlobal = $true; break }
      }
      if(-not $hasGlobal){
        # compute insertion index just after (optional) docstring
        $insIdx = $bodyStart
        # skip blank/comment lines
        while($insIdx -le $funcEnd){
          $t = $lines[$insIdx].Trim()
          if($t -eq '' -or $t -like '#*'){ $insIdx++; continue }
          break
        }
        # if docstring starts here, skip it (tripled quotes, possibly multi-line)
        if($insIdx -le $funcEnd -and $lines[$insIdx].Trim() -match '^(?:"""|\'\'\')'){
          $q = ($lines[$insIdx].Trim().Substring(0,3))
          $insIdx++
          # walk until closing
          while($insIdx -le $funcEnd){
            if($lines[$insIdx] -match [regex]::Escape($q)){ $insIdx++; break }
            $insIdx++
          }
        }

        $insertIndent = ' ' * ($funcIndent + 4)
        $globalLine = "$insertIndent" + 'global ' + ($nvVars -join ', ')
        $lines = $lines[0..($insIdx-1)] + @($globalLine) + $lines[$insIdx..($lines.Count-1)]
        Write-Host ("[insert] global after docstring in function starting line {0}" -f ($funcStart+1))
        # adjust idx to account for insertion
        $idx = $funcStart
      }
    }

    # jump to end
    $idx = $funcEnd + 1
    continue
  }
  $idx++
}

# 4) Write back
$new = ($lines -join "`r`n")
if($new -ne $raw){
  Set-Content -Path $FilePath -Value $new -Encoding UTF8 -NoNewline
  Write-Host "[patched] $FilePath"
}else{
  Write-Host "[no-op] Already compliant."
}
Write-Host "[done]"
