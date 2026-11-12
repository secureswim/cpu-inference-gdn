param([switch]$AllowBattery, [string]$LlamaSource = '')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot/..
if (!(Test-Path .venv/Scripts/python.exe)) { python -m venv .venv }
$py = Join-Path (Get-Location) '.venv/Scripts/python.exe'
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE) { throw 'Dependency installation failed' }
if (!(Test-Path build/vendor/ggml)) {
    if ($LlamaSource) { & $py scripts/prepare_ggml.py --local-source $LlamaSource }
    else { & $py scripts/prepare_ggml.py }
    if ($LASTEXITCODE) { throw 'Source preparation failed' }
}
cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
if ($LASTEXITCODE) { throw 'Configuration failed' }
cmake --build build/cmake -j 6
if ($LASTEXITCODE) { throw 'Compilation failed' }
& build/cmake/gdn.exe check | Set-Content results/gdn_correctness.json
if ($LASTEXITCODE) { throw 'GDN correctness failed' }
& $py -m pip freeze | Set-Content results/python-lock.txt
if ($AllowBattery) { & $py scripts/run_benchmarks.py --allow-battery }
else { & $py scripts/run_benchmarks.py }
if ($LASTEXITCODE) { throw 'Benchmark sweep failed' }
& $py scripts/analyze.py
if ($LASTEXITCODE) { throw 'Analysis failed' }
& $py scripts/make_report.py
if ($LASTEXITCODE) { throw 'Report generation failed' }
