# Run validation/sign_check.py over a list of objectives, one at a time,
# detached, BelowNormal. Each objective appends its line to
# data/sign_check.jsonl when done; objectives already in that file are skipped,
# so a re-launch after a shutdown resumes.
#   powershell -File control\run_sign_check.ps1 -Costs "tts;tts:1|jam:2.5"
# (one ';'-separated string: `-File` does not split a PowerShell array)
param([string]$Costs)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = (Get-Command python).Source
foreach ($c in ($Costs -split ';')) {
    $done = (Test-Path data\sign_check.jsonl) -and
            (Select-String -Path data\sign_check.jsonl -SimpleMatch "`"cost`": `"$c`"" -Quiet)
    if ($done) { continue }
    $tag = ($c -replace '[^A-Za-z0-9_.-]', '_')
    $p = Start-Process -FilePath $py -WindowStyle Hidden -PassThru -WorkingDirectory $root `
        -ArgumentList @('-u', 'validation\sign_check.py', "`"$c`"", '1', '4600') `
        -RedirectStandardOutput "data\sign_check_$tag.log" `
        -RedirectStandardError "data\sign_check_$tag.err"
    $p.PriorityClass = 'BelowNormal'
    $p.WaitForExit()
}
