# TurboInfer Build Script for Windows PowerShell
# This script sets up the environment and builds the project

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "TurboInfer Build Script" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check for required tools
Write-Host "Checking for required tools..." -ForegroundColor Yellow

if (-not (Test-Command "cmake")) {
    Write-Host "Error: CMake not found in PATH" -ForegroundColor Red
    Write-Host "Please make sure CMake is installed and restart PowerShell" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Command "g++")) {
    Write-Host "Error: g++ compiler not found in PATH" -ForegroundColor Red
    Write-Host "Please restart PowerShell to refresh environment variables" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "CMake version:" -ForegroundColor Green
cmake --version

Write-Host ""
Write-Host "GCC version:" -ForegroundColor Green
g++ --version

Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Building TurboInfer..." -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# Create build directory
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Name "build" | Out-Null
}

# Configure with CMake
Write-Host "Configuring project..." -ForegroundColor Yellow
$configResult = & cmake -B build -S . -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=20

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: CMake configuration failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Build the project
Write-Host "Building project..." -ForegroundColor Yellow
$buildResult = & cmake --build build --config Debug --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Build failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan

Write-Host "Running tests..." -ForegroundColor Yellow
Set-Location "build"
& ctest --output-on-failure

Write-Host ""
Write-Host "Build and test completed!" -ForegroundColor Green
Read-Host "Press Enter to exit"
