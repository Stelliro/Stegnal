# fix_function_global_order.ps1
# PS 5.1-safe, surgical:
# - For each function, if it contains `global _NVRTC_*`, move/consolidate that global
#   to the first executable line (after optional docstring), BEFORE any use.
# - Touches nothing else. Backs up once if changes are needed.
param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py",
  [switch]$Verify
)

if(!(Test-Path $FilePath)){ Write-Error "File not found: $FilePath"; exit 1 }

$nvVars = @('_NVRTC_PATH_CACHED','_NVRTC_DETECTED_LIBRARY','_NVRTC_DETECTED_VERSION','_NVRTC_VERSION_MATCHED')
$nvAlt  = ($nvVars -join '|')
$reDef  = '^\s*def\s+\w+\s*\(.*\)\s*:\s*$'
$reGlob = "^\s*global\s+.*($nvAlt)"
$reAssignOrUse = "($nvAlt)"   # any reference is enough to require early global
$reDocOpen = '^\s*(?:"{3}|''{3})'

$raw   = Get-Content $FilePath -Raw -Encoding UTF8
$lines = $raw -split "`r?`n"

function GetIndent([string]$s){ ($s -replace '([^\s]).*$','').Length }

$changes = @()
$idx = 0
while($idx -lt $lines.Count){
  $line = $lines[$idx]
  if($line -match $reDef){
    $fnStart   = $idx
    $fnIndent  = GetIndent $line
    $bodyStart = $fnStart + 1

    # find end of function by indentation drop
    $fnEnd = $lines.Count - 1
    for($j=$bodyStart; $j -lt $lines.Count; $j++){
      $t = $lines[$j].Trim()
      $ind = GetIndent $lines[$j]
      if($t -eq '' -or $t -like '#*'){ continue }
      if($ind -le $fnIndent -and $j -gt $fnStart){ $fnEnd = $j - 1; break }
    }

    # collect any global lines for our vars
    $globIdxs = @()
    $globVars = New-Object System.Collections.Generic.HashSet[string]
    for($k=$bodyStart; $k -le $fnEnd; $k++){
      if($lines[$k] -match $reGlob){
        $globIdxs += $k
        # parse var names from this global line
        $parts = $lines[$k] -replace '^\s*global\s+',''
        $parts = ($parts -split ',') | % { $_.Trim() }
        foreach($p in $parts){ if($nvVars -contains $p){ [void]$globVars.Add($p) } }
      }
    }

    if($globIdxs.Count -gt 0){
      # find first executable line (skip blanks/comments/docstring block if present)
      $ins = $bodyStart
      while($ins -le $fnEnd){
        $t = $lines[$ins].Trim()
        if($t -eq '' -or $t -like '#*'){ $ins++; continue }
        break
      }
      # if a docstring starts here, skip the whole block
      if($ins -le $fnEnd -and $lines[$ins].Trim() -match $reDocOpen){
        $q = $lines[$ins].Trim().Substring(0,3)  # """ or '''
        $ins++
        while($ins -le $fnEnd){
          if($lines[$ins] -match [regex]::Escape($q)){ $ins++; break }
          $ins++
        }
      }

      # We will insert ONE consolidated global line at $ins using all vars mentioned anywhere in the function.
      $targetLine = (' ' * ($fnIndent + 4)) + 'global ' + (($globVars.ToArray() | Sort-Object) -join ', ')
      # If an identical global already exists at $ins, skip; else, remove all existing global lines and insert one.
      $needsMove = $true
      if($ins -le $fnEnd -and ($lines[$ins] -replace '\s+',' ') -eq ($targetLine -replace '\s+',' ')){ $needsMove = $false }

      if($needsMove){
        # Remove existing global lines (bottom-up)
        foreach($gi in ($globIdxs | Sort-Object -Descending)){
          $lines = $lines[0..($gi-1)] + $lines[($gi+1)..($lines.Count-1)]
          if($gi -lt $ins){ $ins-- ; $fnEnd-- }
        }
        # Insert consolidated global
        $lines = $lines[0..($ins-1)] + @($targetLine) + $lines[$ins..($lines.Count-1)]
        $changes += "Moved global -> line $($ins+1) in function starting at line $($fnStart+1)"
      }
    }

    $idx = $fnEnd + 1
    continue
  }
  $idx++
}

if($changes.Count -eq 0){
  Write-Host "[no-op] No function-level global reordering needed."
  if($Verify){ exit 0 }
  exit 0
}

if($Verify){
  Write-Host "Planned moves:" -ForegroundColor Yellow
  $changes | % { Write-Host "  $_" }
  exit 0
}

$bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
Copy-Item $FilePath $bak -Force
Write-Host "[backup] $bak"
($lines -join "`r`n") | Set-Content $FilePath -Encoding UTF8 -NoNewline
Write-Host "[patched] Reordered function-level globals:"
$changes | % { Write-Host "  $_" }
