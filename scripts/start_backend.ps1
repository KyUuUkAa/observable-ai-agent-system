[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [string]$CondaEnv = "",
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [switch]$NoReload,
    [switch]$SkipDependencyCheck,
    [switch]$SkipServiceChecks
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
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

function Resolve-PythonRunner {
    if ($PythonPath -and $CondaEnv) {
        throw "Use either -PythonPath or -CondaEnv, not both."
    }

    if ($PythonPath) {
        if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
            throw "Python executable not found at: $PythonPath"
        }
        return [pscustomobject]@{ Command = (Resolve-Path -LiteralPath $PythonPath).Path; PrefixArgs = @() }
    }

    if ($CondaEnv) {
        $conda = Resolve-Executable -Names @("conda.exe", "conda")
        if (-not $conda) {
            throw "Conda was not found on PATH."
        }
        return [pscustomobject]@{
            Command = $conda
            PrefixArgs = @("run", "--no-capture-output", "-n", $CondaEnv, "python")
        }
    }

    if ($env:CONDA_PREFIX) {
        $activeCondaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $activeCondaPython -PathType Leaf) {
            return [pscustomobject]@{ Command = $activeCondaPython; PrefixArgs = @() }
        }
    }

    $python = Resolve-Executable -Names @("python.exe", "python")
    if ($python) {
        try {
            & $python --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return [pscustomobject]@{ Command = $python; PrefixArgs = @() }
            }
        }
        catch {
            # Fall through to the Python launcher.
        }
    }

    $pyLauncher = Resolve-Executable -Names @("py.exe", "py")
    if ($pyLauncher) {
        return [pscustomobject]@{ Command = $pyLauncher; PrefixArgs = @("-3") }
    }

    throw "No working Python 3 executable was found. Activate Conda or pass -CondaEnv / -PythonPath."
}

if (-not $SkipDependencyCheck) {
    $checkParameters = @{ Scope = "Backend" }
    if ($PythonPath) { $checkParameters.PythonPath = $PythonPath }
    if ($CondaEnv) { $checkParameters.CondaEnv = $CondaEnv }
    if ($SkipServiceChecks) { $checkParameters.SkipServiceChecks = $true }

    & $CheckScript @checkParameters
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$runner = Resolve-PythonRunner
$command = $runner.Command
$arguments = @($runner.PrefixArgs) + @(
    "-m", "uvicorn", "api:app",
    "--host", $HostAddress,
    "--port", $Port.ToString()
)
if (-not $NoReload) {
    $arguments += "--reload"
}

Write-Host ""
Write-Host "Starting backend" -ForegroundColor Cyan
Write-Host "API:     http://${HostAddress}:$Port" -ForegroundColor Green
Write-Host "Health:  http://${HostAddress}:$Port/health" -ForegroundColor Green
Write-Host "Swagger: http://${HostAddress}:$Port/docs" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

Push-Location $ProjectRoot
try {
    & $command @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
