# Run validation/prediction_study.py over every road/demand/seed, one at a
# time, detached, at BelowNormal priority (it shares the machine with the
# round-42 sweep workers). Skips any case whose JSON already exists, so it can
# simply be re-launched after an interruption.
#
#   powershell -File control\run_prediction_study.ps1 [-WaitPid 1234]
param([int]$WaitPid = 0)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if ($WaitPid -gt 0) {
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep 30 }
}
$cases = @(
    @('straight', '1', '4600'), @('straight', '2', '4600'), @('straight', '3', '4600'),
    @('straight', '1', '4800'), @('straight', '2', '4800'), @('straight', '3', '4800'),
    @('long', '1'), @('long', '2'), @('long', '3'))
$py = (Get-Command python).Source
foreach ($c in $cases) {
    $name = if ($c.Count -gt 2) { "$($c[0])_d$($c[2])_s$($c[1])" } else { "$($c[0])_s$($c[1])" }
    if (Test-Path "data\prediction_study\$name.json") { continue }
    $p = Start-Process -FilePath $py -WindowStyle Hidden -PassThru -Wait:$false `
        -WorkingDirectory $root `
        -ArgumentList (@('-u', 'validation\prediction_study.py') + $c) `
        -RedirectStandardOutput "data\prediction_study\$name.log" `
        -RedirectStandardError "data\prediction_study\$name.err"
    $p.PriorityClass = 'BelowNormal'
    $p.WaitForExit()
}
