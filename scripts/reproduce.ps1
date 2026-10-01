# Verify a clean committed checkout and an isolated dependency installation.
# Local artifact hardlinks reuse immutable downloaded files without duplicating GBs.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$verificationRoot = Join-Path $projectRoot ('.repro\' + [guid]::NewGuid().ToString('N'))
$checkout = Join-Path $verificationRoot 'checkout'
$environmentRoot = Join-Path $verificationRoot 'venv'
New-Item -ItemType Directory -Path $verificationRoot -Force | Out-Null
git clone https://github.com/Kaaushikk/on-device-quantized-ai.git $checkout
if ($LASTEXITCODE -ne 0) { throw 'Clean checkout failed' }
& (Join-Path $projectRoot '.venv\Scripts\python.exe') -m venv $environmentRoot
if ($LASTEXITCODE -ne 0) { throw 'Fresh environment creation failed' }
$verificationPython = Join-Path $environmentRoot 'Scripts\python.exe'
& $verificationPython -m pip install -r (Join-Path $checkout 'requirements.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Locked installation failed' }
& $verificationPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
# Link only the model, converted graphs, data, and manifests. Do not link caches
# or failed preprocessing diagnostics. This check never exports or edits weights.
$artifactSource = Join-Path $projectRoot 'artifacts'
$artifactDestination = Join-Path $checkout 'artifacts'
$selectedFiles = @('data.json', 'pytorch_fp32.manifest.json', 'onnx_fp32.manifest.json',
                   'onnx_int8.manifest.json', 'model.fp32.onnx', 'model.int8.onnx')
foreach ($name in $selectedFiles) {
    New-Item -ItemType HardLink -Path (Join-Path $artifactDestination $name) -Target (Join-Path $artifactSource $name) | Out-Null
}
New-Item -ItemType Directory -Path (Join-Path $artifactDestination 'model') -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $artifactSource 'model') -File | ForEach-Object {
    New-Item -ItemType HardLink -Path (Join-Path (Join-Path $artifactDestination 'model') $_.Name) -Target $_.FullName | Out-Null
}
$env:PYTHONPATH = Join-Path $checkout 'src'
Push-Location $checkout
try {
    & $verificationPython -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Clean checkout tests failed' }
    & $verificationPython scripts\check_offline.py
    if ($LASTEXITCODE -ne 0) { throw 'Offline checks failed in fresh environment' }
    & $verificationPython -m quantbench.cli parity
    if ($LASTEXITCODE -ne 0) { throw 'Parity failed in fresh environment' }
    $commit = git rev-parse HEAD
    $record = [ordered]@{
        passed = $true
        commit = $commit
        method = 'Clean public GitHub checkout; newly installed locked environment; hash-verified local artifact hardlinks'
        checks = @('dependency compatibility', 'unit tests', 'offline all-variant inference', 'FP32 parity')
        limitations = @('Reuses downloaded source artifacts', 'Does not repeat the full device timing protocol', 'Python socket block is not an OS firewall test')
    }
    $record | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 (Join-Path $projectRoot 'results\reproduction.json')
} finally {
    Pop-Location
}
Write-Output 'Clean-checkout and fresh-environment verification passed.'
