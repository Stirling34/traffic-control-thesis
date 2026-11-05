# Launch N sweep workers fully detached from the calling shell, so they
# survive the terminal (or an agent session) closing. Windows PowerShell 5.1.
#
#   powershell -File control\launch_sweep.ps1 -Round final -Dir data\sweep_final -N 6
#
# Guards, each learned the hard way (docs/log.md ledger):
#   * refuses to start if ANY sweep worker is already running (one writer per
#     shard; a stale process once clobbered a finished run, 2026-08-21);
#   * counts python.exe only (nohup/cmd wrappers doubled the count before);
#   * at most 6 workers on the laptop (page-file growth, 2026-10-05).
# Workers run at BelowNormal priority: no throughput lost when the machine is
# otherwise idle, and it stays usable if someone opens an app.
param(
    [Parameter(Mandatory = $true)][string]$Round,
    [Parameter(Mandatory = $true)][string]$Dir,
    [int]$N = 6
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if ($N -gt 6) { throw "at most 6 workers on this machine (ledger 2026-10-05)" }

$running = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'sweep\.py --worker' })
if ($running.Count -gt 0) {
    throw "$($running.Count) sweep worker(s) already running; stop them first"
}

python control\sweep.py --manifest --round $Round --dir $Dir
if ($LASTEXITCODE -ne 0) { throw "manifest build failed" }

$py = (Get-Command python).Source
for ($k = 0; $k -lt $N; $k++) {
    $log = Join-Path $root "$Dir\${Round}_w$k.log"
    $err = Join-Path $root "$Dir\${Round}_w$k.err"
    $p = Start-Process -FilePath $py -WindowStyle Hidden -PassThru `
        -WorkingDirectory $root `
        -ArgumentList @('-u', 'control\sweep.py', '--worker', "$k",
                        '--workers', "$N", '--dir', $Dir) `
        -RedirectStandardOutput $log -RedirectStandardError $err
    $p.PriorityClass = 'BelowNormal'
    "worker $k pid $($p.Id) -> $log"
}
Start-Sleep -Seconds 15
$now = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'sweep\.py --worker' })
"$($now.Count) workers running at $(Get-Date)"
