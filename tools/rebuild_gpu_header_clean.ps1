# rebuild_gpu_header_clean.ps1
# PS 5.1-compatible. Rebuilds gpu_runtime.py header safely:
# - Preserves pre-import comments/docstrings
# - Preserves the initial contiguous import cluster
# - Comments out any stray 'try:' before the stub block
# - Injects clean NumPy/CuPy stubs
# - Removes any prior/broken stubs and stray 'import cupy as cp'

param(
  [string]$FilePath = "D:\STELLIROS_WORKSHOP\TOP SECRET\Project-Umbra\src\umbra\gpu_runtime.py"
)

if (!(Test-Path $FilePath)) { Write-Error "File not found: $FilePath"; exit 1 }

# Backup
$bak = "$FilePath.$((Get-Date).ToString('yyyyMMdd_HHmmss')).bak"
Copy-Item $FilePath $bak -Force
Write-Host "[backup] $bak"

# Load & normalize tabs
$raw  = Get-Content $FilePath -Raw -Encoding UTF8
$norm = $raw -replace "`t","    "
$lines = $norm -split "`r?`n"

# 1) Identify header sections
$firstImport = -1
$endImports  = -1

for($i=0;$i -lt $lines.Count;$i++){
  if($lines[$i] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $firstImport = $i; break }
}

if($firstImport -ge 0){
  $endImports = $firstImport
  for($j=$firstImport; $j -lt $lines.Count; $j++){
    if($lines[$j] -match '^\s*(import\s+|from\s+\S+\s+import\s+)'){ $endImports = $j; continue }
    if($lines[$j].Trim() -eq ''){ $endImports = $j; continue }  # allow blank spacer within cluster
    break
  }
} else {
  # no imports found; treat whole top as "pre" and inject stubs at top
  $firstImport = $lines.Count
  $endImports  = $lines.Count - 1
}

$pre   = @()
$imps  = @()
$post  = @()

if($firstImport -gt 0){ $pre  = $lines[0..($firstImport-1)] }
if($firstImport -le $endImports -and $firstImport -ge 0){ $imps = $lines[$firstImport..$endImports] }
if($endImports + 1 -lt $lines.Count){ $post = $lines[($endImports+1)..($lines.Count-1)] }

# 2) Comment out any dangling 'try:' in the pre+import area (before stubs)
function CommentDanglingTry([string[]]$arr){
  $out = New-Object System.Collections.Generic.List[string]
  $openTry = -1
  for($k=0;$k -lt $arr.Count;$k++){
    $t = $arr[$k].Trim()
    if($t -like 'try:*'){
      # open a try; tentatively comment it, unless we see an except/finally later before def/class
      $openTry = $k
      $out.Add("# FIX: neutralized dangling $t")
      continue
    }
    if($t -like 'except*' -or $t -like 'finally*'){
      $openTry = -1
      $out.Add($arr[$k])
      continue
    }
    if(($t -like 'def *' -or $t -like 'class *') -and $openTry -ge 0){
      # we already commented the try; just continue
      $openTry = -1
      $out.Add($arr[$k])
      continue
    }
    $out.Add($arr[$k])
  }
  return ,$out.ToArray()
}

$pre  = CommentDanglingTry $pre
$imps = CommentDanglingTry $imps

# 3) Build clean NumPy/CuPy stubs
$stub = @(
  ""
  "try:  # numpy-import-stub"
  "    import numpy as _np"
  "except Exception:"
  "    _np = None"
  ""
  "# cupy-import-stub"
  "try:"
  "    import cupy as cp"
  "except Exception:"
  "    cp = None"
  ""
)

# 4) Scrub any previous/broken stubs and stray 'import cupy as cp' from the post section
$cleanPost = New-Object System.Collections.Generic.List[string]
$skipping = $false
$mode = ""
for($i=0;$i -lt $post.Count;$i++){
  $ln = $post[$i]
  if(-not $skipping -and $ln -match '^\s*try:\s*#\s*numpy-import-stub'){ $skipping=$true; $mode="numpy"; continue }
  if(-not $skipping -and $ln -match '^\s*#\s*cupy-import-stub'){        $skipping=$true; $mode="cupy";  continue }
  if($skipping){
    if($ln.Trim() -eq ''){ $skipping=$false; $mode=""; continue } else { continue }
  }
  if($ln -match '^\s*import\s+cupy\s+as\s+cp\s*$'){ continue }
  $cleanPost.Add($ln)
}
$post = $cleanPost.ToArray()

# 5) Reassemble file: pre + imports + stub + post
$final = @()
$final += $pre
if($imps.Count -gt 0){ $final += $imps }
# ensure one blank line between imports and stub
if($final.Count -gt 0 -and ($final[-1]).Trim() -ne ''){ $final += @('') }
$final += $stub
# ensure a blank line after stub if needed
if($post.Count -gt 0 -and $post[0].Trim() -ne ''){ $final += @('') }
$final += $post

# 6) Write back with CRLF
$joined = ($final -join "`r`n")
Set-Content -Path $FilePath -Value $joined -Encoding UTF8 -NoNewline
Write-Host "[patched] rebuilt header and injected clean stubs"
