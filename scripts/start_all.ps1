[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [string]$CondaEnv = "",
    [string]$BackendHost = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$BackendPort = 8000,
    [string]$FrontendHost = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$FrontendPort = 5173,
    [switch]$NoReload,
    [switch]$SkipServiceChecks
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$CheckScript = Join-Path $ScriptDir "check_environment.ps1"
$BackendScript = Join-Path $ScriptDir "start_backend.ps1"
$FrontendScript = Join-Path $ScriptDir "start_frontend.ps1"
$FrontendModules = Join-Path $ProjectRoot "XYNai-agent\node_modules"

if ($PythonPath -and $CondaEnv) {
    Write-Host "[FAIL] Use either -PythonPath or -CondaEnv, not both." -ForegroundColor Red
    exit 1
}

$checkParameters = @{ Scope = "All" }
if ($PythonPath) { $checkParameters.PythonPath = $PythonPath }
if ($CondaEnv) { $checkParameters.CondaEnv = $CondaEnv }
if ($SkipServiceChecks) { $checkParameters.SkipServiceChecks = $true }

& $CheckScript @checkParameters
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath $FrontendModules -PathType Container)) {
    Write-Host "[FAIL] Frontend dependencies are missing." -ForegroundColor Red
    Write-Host "Run: Set-Location XYNai-agent; npm install" -ForegroundColor Yellow
    exit 1
}

$powerShellExe = (Get-Process -Id $PID).Path
if (-not $powerShellExe) {
    $powerShellExe = (Get-Command "powershell.exe" -ErrorAction Stop).Source
}

function Quote-Argument {
    param([string]$Value)
    return '"' + $Value.Replace('"', '\"') + '"'
}

$backendArguments = @(
    "-NoExit",
    "-File", (Quote-Argument $BackendScript),
    "-HostAddress", (Quote-Argument $BackendHost),
    "-Port", $BackendPort,
    "-SkipDependencyCheck"
)
if ($PythonPath) { $backendArguments += @("-PythonPath", (Quote-Argument $PythonPath)) }
if ($CondaEnv) { $backendArguments += @("-CondaEnv", (Quote-Argument $CondaEnv)) }
if ($NoReload) { $backendArguments += "-NoReload" }
if ($SkipServiceChecks) { $backendArguments += "-SkipServiceChecks" }

$frontendArguments = @(
    "-NoExit",
    "-File", (Quote-Argument $FrontendScript),
    "-HostAddress", (Quote-Argument $FrontendHost),
    "-Port", $FrontendPort,
    "-SkipDependencyCheck"
)

$backendProcess = Start-Process -FilePath $powerShellExe -ArgumentList $backendArguments -WorkingDirectory $ProjectRoot -PassThru
$frontendProcess = Start-Process -FilePath $powerShellExe -ArgumentList $frontendArguments -WorkingDirectory $ProjectRoot -PassThru

Write-Host ""
Write-Host "Backend and frontend were launched in separate PowerShell windows." -ForegroundColor Green
Write-Host "Backend PID: $($backendProcess.Id) - http://${BackendHost}:$BackendPort" -ForegroundColor Cyan
Write-Host "Frontend PID: $($frontendProcess.Id) - http://${FrontendHost}:$FrontendPort" -ForegroundColor Cyan
Write-Host "Close those windows or press Ctrl+C in each window to stop the services." -ForegroundColor Yellow
