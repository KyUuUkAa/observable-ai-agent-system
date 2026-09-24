[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 5173,
    [switch]$SkipDependencyCheck
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$FrontendDir = Join-Path $ProjectRoot "XYNai-agent"
$CheckScript = Join-Path $ScriptDir "check_environment.ps1"

function Resolve-Executable {
    param([string[]]$Names)
    foreach ($name in $Names) {
        $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) {
            return $command.Source
        }
    }
    return $null
}

if (-not $SkipDependencyCheck) {
    & $CheckScript -Scope Frontend
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$nodeModules = Join-Path $FrontendDir "node_modules"
if (-not (Test-Path -LiteralPath $nodeModules -PathType Container)) {
    Write-Host "[FAIL] Frontend dependencies are missing." -ForegroundColor Red
    Write-Host "Run: Set-Location XYNai-agent; npm install" -ForegroundColor Yellow
    exit 1
}

$npm = Resolve-Executable -Names @("npm.cmd", "npm.exe", "npm")
if (-not $npm) {
    Write-Host "[FAIL] npm was not found. Install Node.js with npm included." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting frontend" -ForegroundColor Cyan
Write-Host "URL: http://${HostAddress}:$Port" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

Push-Location $FrontendDir
try {
    & $npm run dev -- --host $HostAddress --port $Port
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
