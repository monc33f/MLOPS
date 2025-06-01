# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Restarting script with administrator privileges..."
    Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    Exit
}

# Ensure we're in the correct directory
Set-Location "D:\PFA 2A\projet"

# Activate virtual environment
if (Test-Path ".\MLOPS1\Scripts\Activate.ps1") {
    . .\MLOPS1\Scripts\Activate.ps1
} else {
    Write-Error "Virtual environment not found. Please ensure MLOPS1 environment exists."
    Exit 1
}

# Check if Docker Desktop is running
$dockerProcess = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
if (-not $dockerProcess) {
    Write-Host "Starting Docker Desktop..."
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    Write-Host "Waiting for Docker to start (this may take a minute)..."
    Start-Sleep -Seconds 60
}

# Wait for Docker daemon to be ready
$retries = 0
$maxRetries = 5
$ready = $false

while (-not $ready -and $retries -lt $maxRetries) {
    try {
        $null = docker info
        $ready = $true
        Write-Host "Docker is ready!"
    }
    catch {
        $retries++
        Write-Host "Waiting for Docker to be ready... (Attempt $retries of $maxRetries)"
        Start-Sleep -Seconds 10
    }
}

if (-not $ready) {
    Write-Error "Docker failed to start properly. Please start Docker Desktop manually and try again."
    Exit 1
}

# Run docker-compose
Write-Host "Starting containers..."
docker-compose up --build

# Keep the window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} 