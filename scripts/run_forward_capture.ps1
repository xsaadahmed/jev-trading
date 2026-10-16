# Daily Phase 13A forward INPUT capture (no orders). Runs after 18:30 ET gate.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
if (Test-Path .\.env) {
  Get-Content .\.env | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $k,$v = $_.Split('=',2)
    if ($null -ne $k -and $null -ne $v) {
      Set-Item -Path "Env:$($k.Trim())" -Value $v.Trim().Trim('"').Trim("'")
    }
  }
}
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $py -m jevlab.cli.main run forward-capture
exit $LASTEXITCODE
