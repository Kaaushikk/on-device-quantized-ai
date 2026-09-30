param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
$projectRoot = Split-Path $PSScriptRoot -Parent
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create .venv and install requirements.lock.txt first. See docs/REPRODUCE.md.'
}
$env:PYTHONPATH = Join-Path $projectRoot 'src'
& $projectPython -m quantbench.cli @Arguments
exit $LASTEXITCODE
