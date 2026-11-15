param([switch]$AllowBattery, [switch]$LongRepeats, [string]$Output = 'build/reproduction')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot/..
$py = '.venv/Scripts/python.exe'
if (!(Test-Path $py)) {
    throw 'Run python scripts/configure.py first.'
}
$options = @('--output', $Output)
if ($AllowBattery) { $options += '--allow-battery' }
if ($LongRepeats) { $options += '--long-repeats' }
& $py scripts/run.py @options
if ($LASTEXITCODE) { throw 'Workflow failed. See the error above.' }
