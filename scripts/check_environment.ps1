[CmdletBinding()]
param(
    [ValidateSet("All", "Backend", "Frontend")]
    [string]$Scope = "All",

    [string]$PythonPath = "",

    [string]$CondaEnv = "",

    [switch]$SkipServiceChecks
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$FrontendDir = Join-Path $ProjectRoot "XYNai-agent"
$script:FailureCount = 0

function Write-Result {
    param(
        [ValidateSet("OK", "INFO", "WARN", "FAIL")]
        [string]$Level,
        [string]$Message
    )

    $color = switch ($Level) {
        "OK" { "Green" }
        "INFO" { "Cyan" }
        "WARN" { "Yellow" }
        "FAIL" { "Red" }
    }

    Write-Host ("[{0,-4}] {1}" -f $Level, $Message) -ForegroundColor $color
}

function Add-Failure {
    param([string]$Message)
    $script:FailureCount++
    Write-Result -Level "FAIL" -Message $Message
}

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

function Invoke-Runner {
    param(
        [pscustomobject]$Runner,
        [string[]]$Arguments
    )

    $command = $Runner.Command
    $allArguments = @($Runner.PrefixArgs) + $Arguments
    $output = & $command @allArguments 2>&1

    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output = (($output | Out-String).Trim())
    }
}

function Resolve-PythonRunner {
    if ($PythonPath -and $CondaEnv) {
        Add-Failure "Use either -PythonPath or -CondaEnv, not both."
        return $null
    }

    if ($PythonPath) {
        if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
            Add-Failure "Python executable not found at: $PythonPath"
            return $null
        }

        $runner = [pscustomobject]@{
            Command = (Resolve-Path -LiteralPath $PythonPath).Path
            PrefixArgs = @()
            Description = "PythonPath"
        }
        try {
            $probe = Invoke-Runner -Runner $runner -Arguments @("--version")
            if ($probe.ExitCode -eq 0) {
                Write-Result -Level "OK" -Message ("Python: {0} ({1})" -f $probe.Output, $runner.Command)
                return $runner
            }
        }
        catch {
            Add-Failure "The configured Python executable could not run: $($_.Exception.Message)"
            return $null
        }
    }

    if ($CondaEnv) {
        $conda = Resolve-Executable -Names @("conda.exe", "conda")
        if (-not $conda) {
            Add-Failure "Conda was requested for environment '$CondaEnv', but conda was not found on PATH."
            return $null
        }

        $runner = [pscustomobject]@{
            Command = $conda
            PrefixArgs = @("run", "--no-capture-output", "-n", $CondaEnv, "python")
            Description = "conda:$CondaEnv"
        }
        try {
            $probe = Invoke-Runner -Runner $runner -Arguments @("--version")
            if ($probe.ExitCode -eq 0) {
                # Resolve the environment's actual Python executable once.
                # Passing a quote-heavy `python -c` payload through `conda run`
                # is unreliable in Windows PowerShell 5.1.
                $pathProbe = Invoke-Runner -Runner $runner -Arguments @(
                    "-c",
                    "import sys; print(sys.executable)"
                )
                if ($pathProbe.ExitCode -eq 0) {
                    $resolvedPython = $pathProbe.Output.Trim()
                    if (Test-Path -LiteralPath $resolvedPython -PathType Leaf) {
                        Write-Result -Level "OK" -Message ("Python: {0} (Conda environment: {1})" -f $probe.Output, $CondaEnv)
                        return [pscustomobject]@{
                            Command = (Resolve-Path -LiteralPath $resolvedPython).Path
                            PrefixArgs = @()
                            Description = "conda:$CondaEnv"
                        }
                    }
                }

                Add-Failure "Conda environment '$CondaEnv' runs, but its Python executable could not be resolved."
                return $null
            }

            Add-Failure "Conda environment '$CondaEnv' is missing or cannot run Python. Create it first or pass a valid -CondaEnv."
            return $null
        }
        catch {
            Add-Failure "Conda environment '$CondaEnv' could not run Python: $($_.Exception.Message)"
            return $null
        }
    }

    $candidates = @()
    if ($env:CONDA_PREFIX) {
        $activeCondaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $activeCondaPython -PathType Leaf) {
            $candidates += [pscustomobject]@{
                Command = $activeCondaPython
                PrefixArgs = @()
                Description = "active Conda environment"
            }
        }
    }

    foreach ($name in @("python.exe", "python")) {
        $resolved = Resolve-Executable -Names @($name)
        if ($resolved -and -not ($candidates.Command -contains $resolved)) {
            $candidates += [pscustomobject]@{
                Command = $resolved
                PrefixArgs = @()
                Description = "PATH"
            }
        }
    }

    $pyLauncher = Resolve-Executable -Names @("py.exe", "py")
    if ($pyLauncher) {
        $candidates += [pscustomobject]@{
            Command = $pyLauncher
            PrefixArgs = @("-3")
            Description = "Python launcher"
        }
    }

    foreach ($candidate in $candidates) {
        try {
            $probe = Invoke-Runner -Runner $candidate -Arguments @("--version")
            if ($probe.ExitCode -eq 0) {
                Write-Result -Level "OK" -Message ("Python: {0} ({1})" -f $probe.Output, $candidate.Description)
                return $candidate
            }
        }
        catch {
            # Try the next candidate. This commonly skips the Windows Store alias.
        }
    }

    Add-Failure "A working Python 3 executable was not found. Activate the Conda environment, or pass -CondaEnv <name> / -PythonPath <path>."
    return $null
}

function Read-EnvFile {
    param([string]$Path)

    $settings = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
            $value = $Matches[2].Trim()
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            $settings[$Matches[1]] = $value
        }
    }

    return $settings
}

function Test-TcpPort {
    param(
        [string]$ComputerName,
        [int]$Port,
        [int]$TimeoutMilliseconds = 2500
    )

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync($ComputerName, $Port)
        if (-not $task.Wait($TimeoutMilliseconds)) {
            return $false
        }
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Test-BackendEnvironment {
    param([pscustomobject]$PythonRunner)

    Write-Host ""
    Write-Host "Backend checks" -ForegroundColor White

    foreach ($relativePath in @("api.py", "requirements.txt")) {
        $path = Join-Path $ProjectRoot $relativePath
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            Write-Result -Level "OK" -Message "$relativePath found."
        }
        else {
            Add-Failure "$relativePath is missing from the project root."
        }
    }

    $envFile = Join-Path $ProjectRoot ".env"
    $settings = $null
    if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
        Add-Failure ".env is missing. Run: Copy-Item .env.example .env, then replace every placeholder locally."
    }
    else {
        Write-Result -Level "OK" -Message ".env found (values are not displayed)."
        $settings = Read-EnvFile -Path $envFile
        foreach ($key in @("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "AGENT_DATABASE_URL")) {
            if (-not $settings.ContainsKey($key) -or [string]::IsNullOrWhiteSpace($settings[$key])) {
                Add-Failure ".env variable '$key' is missing or empty."
            }
            elseif ($settings[$key] -match '(?i)your[_-]|replace[_-]|change[_-]?me|<.+>') {
                Add-Failure ".env variable '$key' still contains a placeholder."
            }
        }

        if ($settings.ContainsKey("AGENT_DATABASE_URL") -and
            -not $settings["AGENT_DATABASE_URL"].StartsWith("postgresql+asyncpg://")) {
            Add-Failure "AGENT_DATABASE_URL must start with postgresql+asyncpg://."
        }
    }

    $resumeFile = Join-Path $ProjectRoot "data\resume.txt"
    if (Test-Path -LiteralPath $resumeFile -PathType Leaf) {
        Write-Result -Level "OK" -Message "Private RAG file data/resume.txt found."
    }
    else {
        Add-Failure "data/resume.txt is missing. Copy data/resume.example.txt and replace it with local private content."
    }

    $oracleModelSetting = "models/oracle/best_portable.pt"
    if ($settings -and $settings.ContainsKey("ORACLE_MODEL_PATH") -and
        -not [string]::IsNullOrWhiteSpace($settings["ORACLE_MODEL_PATH"])) {
        $oracleModelSetting = $settings["ORACLE_MODEL_PATH"]
    }
    $oracleModelPath = if ([System.IO.Path]::IsPathRooted($oracleModelSetting)) {
        $oracleModelSetting
    }
    else {
        Join-Path $ProjectRoot $oracleModelSetting
    }
    if (Test-Path -LiteralPath $oracleModelPath -PathType Leaf) {
        Write-Result -Level "OK" -Message "Oracle classifier weight found."
    }
    else {
        Add-Failure "Oracle classifier weight is missing. Copy best_portable.pt to models/oracle or set ORACLE_MODEL_PATH in .env."
    }

    if ($PythonRunner) {
        try {
            $importProbe = Invoke-Runner -Runner $PythonRunner -Arguments @(
                "-c",
                "import fastapi, uvicorn, psycopg, asyncpg, sqlalchemy, numpy, sentence_transformers, dotenv, agents, PIL, multipart, ultralytics"
            )
            if ($importProbe.ExitCode -eq 0) {
                Write-Result -Level "OK" -Message "Required Python packages can be imported."
            }
            else {
                Add-Failure "Python dependencies are incomplete. Run: python -m pip install -r requirements.txt"
            }
        }
        catch {
            Add-Failure "Python dependency check failed: $($_.Exception.Message)"
        }
    }

    $postgresClient = Resolve-Executable -Names @("pg_isready.exe", "pg_isready", "psql.exe", "psql")
    if ($postgresClient) {
        Write-Result -Level "OK" -Message "PostgreSQL client tools found."
    }
    else {
        Write-Result -Level "WARN" -Message "PostgreSQL client tools are not on PATH. Runtime can still work through psycopg if the server is reachable."
    }

    if (-not $SkipServiceChecks -and $settings) {
        $postgresHost = $settings["POSTGRES_HOST"]
        $postgresPort = 0
        if (-not [int]::TryParse($settings["POSTGRES_PORT"], [ref]$postgresPort)) {
            Add-Failure "POSTGRES_PORT must be an integer."
        }
        elseif (Test-TcpPort -ComputerName $postgresHost -Port $postgresPort) {
            Write-Result -Level "OK" -Message "PostgreSQL is reachable at ${postgresHost}:$postgresPort."

            if ($PythonRunner -and $script:FailureCount -eq 0) {
                $previousEnvFile = $env:AGENT_ENV_FILE
                $databaseProbePath = $null
                try {
                    $env:AGENT_ENV_FILE = $envFile
                    # Windows PowerShell 5.1 can corrupt quote-heavy arguments
                    # passed to native `python -c`. Execute a temporary script
                    # instead so the same check works in both PowerShell editions.
                    $databaseProbeCode = @'
import os

from dotenv import load_dotenv
import psycopg

load_dotenv(os.environ["AGENT_ENV_FILE"], override=True)
required = {"conversations", "messages", "agent_sessions", "agent_messages"}

with psycopg.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    connect_timeout=3,
) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        present = {row[0] for row in cursor.fetchall()}

missing = required - present
if missing:
    raise RuntimeError(
        "Missing required database tables: " + ", ".join(sorted(missing))
    )
'@
                    $databaseProbePath = [System.IO.Path]::Combine(
                        [System.IO.Path]::GetTempPath(),
                        "observable-agent-db-probe-$PID.py"
                    )
                    $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
                    [System.IO.File]::WriteAllText(
                        $databaseProbePath,
                        $databaseProbeCode,
                        $utf8WithoutBom
                    )
                    $databaseProbe = Invoke-Runner -Runner $PythonRunner -Arguments @($databaseProbePath)
                    if ($databaseProbe.ExitCode -eq 0) {
                        Write-Result -Level "OK" -Message "PostgreSQL credentials and required tables are valid."
                    }
                    else {
                        Add-Failure "PostgreSQL authentication/schema check failed. Create the database and run scripts/init_database.sql."
                    }
                }
                catch {
                    Add-Failure "PostgreSQL authentication/schema check failed: $($_.Exception.Message)"
                }
                finally {
                    if ($databaseProbePath -and (Test-Path -LiteralPath $databaseProbePath -PathType Leaf)) {
                        Remove-Item -LiteralPath $databaseProbePath -Force -ErrorAction SilentlyContinue
                    }
                    if ($null -eq $previousEnvFile) {
                        Remove-Item Env:AGENT_ENV_FILE -ErrorAction SilentlyContinue
                    }
                    else {
                        $env:AGENT_ENV_FILE = $previousEnvFile
                    }
                }
            }
        }
        else {
            Add-Failure "PostgreSQL is not reachable at ${postgresHost}:$postgresPort. Start PostgreSQL and verify .env."
        }
    }

    $ollama = Resolve-Executable -Names @("ollama.exe", "ollama")
    if ($ollama) {
        try {
            $ollamaVersion = & $ollama --version 2>&1
            Write-Result -Level "OK" -Message ("Ollama CLI: {0}" -f (($ollamaVersion | Out-String).Trim()))
        }
        catch {
            Add-Failure "Ollama CLI was found but could not run."
        }
    }
    else {
        Add-Failure "Ollama CLI was not found. Install Ollama and run: ollama pull qwen3:4b"
    }

    if (-not $SkipServiceChecks) {
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 4
            Write-Result -Level "OK" -Message "Ollama service is reachable at 127.0.0.1:11434."
            $modelNames = @($response.models | ForEach-Object { $_.name })
            if ($modelNames -contains "qwen3:4b" -or $modelNames -contains "qwen3:4b-latest") {
                Write-Result -Level "OK" -Message "Ollama model qwen3:4b is installed."
            }
            else {
                Add-Failure "Ollama model qwen3:4b is missing. Run: ollama pull qwen3:4b"
            }
        }
        catch {
            Add-Failure "Ollama service is not reachable at 127.0.0.1:11434. Start Ollama and retry."
        }
    }
}

function Test-FrontendEnvironment {
    Write-Host ""
    Write-Host "Frontend checks" -ForegroundColor White

    foreach ($relativePath in @("package.json", "package-lock.json")) {
        $path = Join-Path $FrontendDir $relativePath
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            Write-Result -Level "OK" -Message "XYNai-agent/$relativePath found."
        }
        else {
            Add-Failure "XYNai-agent/$relativePath is missing."
        }
    }

    $node = Resolve-Executable -Names @("node.exe", "node")
    if ($node) {
        try {
            $nodeVersion = & $node --version 2>&1
            Write-Result -Level "OK" -Message ("Node.js: {0}" -f (($nodeVersion | Out-String).Trim()))
        }
        catch {
            Add-Failure "Node.js was found but could not run."
        }
    }
    else {
        Add-Failure "Node.js was not found. Install a version allowed by XYNai-agent/package.json."
    }

    $npm = Resolve-Executable -Names @("npm.cmd", "npm.exe", "npm")
    if ($npm) {
        try {
            $npmVersion = & $npm --version 2>&1
            Write-Result -Level "OK" -Message ("npm: {0}" -f (($npmVersion | Out-String).Trim()))
        }
        catch {
            Add-Failure "npm was found but could not run."
        }
    }
    else {
        Add-Failure "npm was not found. Reinstall Node.js with npm included."
    }

    $nodeModules = Join-Path $FrontendDir "node_modules"
    if (Test-Path -LiteralPath $nodeModules -PathType Container) {
        Write-Result -Level "OK" -Message "Frontend dependencies are installed."
    }
    else {
        Write-Result -Level "WARN" -Message "Frontend dependencies are not installed. Run: Set-Location XYNai-agent; npm install"
    }
}

Write-Host "Observable AI Agent System - environment check" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot" -ForegroundColor DarkGray
Write-Host "Scope:   $Scope" -ForegroundColor DarkGray

$pythonRunner = $null
if ($Scope -in @("All", "Backend")) {
    $pythonRunner = Resolve-PythonRunner
    Test-BackendEnvironment -PythonRunner $pythonRunner
}

if ($Scope -in @("All", "Frontend")) {
    Test-FrontendEnvironment
}

Write-Host ""
if ($script:FailureCount -gt 0) {
    Write-Result -Level "FAIL" -Message "$($script:FailureCount) required check(s) failed. Fix the messages above and run this script again."
    exit 1
}

if ($SkipServiceChecks -and $Scope -in @("All", "Backend")) {
    Write-Result -Level "WARN" -Message "Service connectivity checks were skipped."
}

Write-Result -Level "OK" -Message "All required checks passed."
exit 0
