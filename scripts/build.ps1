# TurboInfer Build Script for Windows
# Usage: .\build.ps1 [Release|Debug] [options]

param(
    [string]$BuildType = "Release",
    [switch]$Clean = $false,
    [switch]$Tests = $true,
    [switch]$Examples = $true,
    [switch]$Benchmarks = $false,
    [switch]$Install = $false,
    [switch]$Verbose = $false,
    [string]$Generator = "Visual Studio 17 2022",
    [string]$BuildDir = "build"
)

# Colors for output
$Red = "`e[31m"
$Green = "`e[32m"
$Yellow = "`e[33m"
$Blue = "`e[34m"
$Reset = "`e[0m"

function Write-ColoredOutput {
    param([string]$Message, [string]$Color = $Reset)
    Write-Host "$Color$Message$Reset"
}

function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Header
Write-ColoredOutput "=======================================" $Blue
Write-ColoredOutput "    TurboInfer Build Script v1.0" $Blue
Write-ColoredOutput "=======================================" $Blue
Write-Host ""

# Check prerequisites
Write-ColoredOutput "Checking prerequisites..." $Yellow

if (!(Test-Command "cmake")) {
    Write-ColoredOutput "ERROR: CMake not found. Please install CMake and add it to PATH." $Red
    exit 1
}

if (!(Test-Command "git")) {
    Write-ColoredOutput "WARNING: Git not found. Some features may not work." $Yellow
}

$cmakeVersion = cmake --version | Select-String "version" | ForEach-Object { $_.ToString().Split()[2] }
Write-ColoredOutput "✓ CMake version: $cmakeVersion" $Green

# Project root directory
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-ColoredOutput "Project root: $ProjectRoot" $Blue
Write-ColoredOutput "Build type: $BuildType" $Blue
Write-ColoredOutput "Build directory: $BuildDir" $Blue

# Clean build directory if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-ColoredOutput "Cleaning build directory..." $Yellow
    Remove-Item -Recurse -Force $BuildDir
}

# Create build directory
if (!(Test-Path $BuildDir)) {
    Write-ColoredOutput "Creating build directory..." $Yellow
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

Set-Location $BuildDir

# Configure CMake options
$cmakeArgs = @(
    ".."
    "-G", $Generator
    "-DCMAKE_BUILD_TYPE=$BuildType"
    "-DTURBOINFER_BUILD_TESTS=$($Tests.ToString().ToLower())"
    "-DTURBOINFER_BUILD_EXAMPLES=$($Examples.ToString().ToLower())"
    "-DTURBOINFER_BUILD_BENCHMARKS=$($Benchmarks.ToString().ToLower())"
)

if ($Verbose) {
    $cmakeArgs += "--verbose"
}

# Run CMake configuration
Write-ColoredOutput "Configuring project with CMake..." $Yellow
Write-ColoredOutput "Command: cmake $($cmakeArgs -join ' ')" $Blue

try {
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }
    Write-ColoredOutput "✓ Configuration successful" $Green
} catch {
    Write-ColoredOutput "ERROR: CMake configuration failed: $_" $Red
    exit 1
}

# Build the project
Write-ColoredOutput "Building project..." $Yellow

$buildArgs = @(
    "--build", "."
    "--config", $BuildType
)

if ($Verbose) {
    $buildArgs += "--verbose"
}

# Detect CPU count for parallel builds
$cpuCount = (Get-CimInstance -ClassName Win32_ComputerSystem).NumberOfLogicalProcessors
$buildArgs += "--parallel", $cpuCount

Write-ColoredOutput "Building with $cpuCount parallel jobs..." $Blue
Write-ColoredOutput "Command: cmake $($buildArgs -join ' ')" $Blue

$buildStartTime = Get-Date

try {
    & cmake @buildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
    
    $buildEndTime = Get-Date
    $buildDuration = ($buildEndTime - $buildStartTime).TotalSeconds
    Write-ColoredOutput "✓ Build successful (${buildDuration:F1}s)" $Green
} catch {
    Write-ColoredOutput "ERROR: Build failed: $_" $Red
    exit 1
}

# Run tests if enabled
if ($Tests) {
    Write-ColoredOutput "Running tests..." $Yellow
    
    try {
        & ctest --build-config $BuildType --output-on-failure
        if ($LASTEXITCODE -ne 0) {
            Write-ColoredOutput "WARNING: Some tests failed" $Yellow
        } else {
            Write-ColoredOutput "✓ All tests passed" $Green
        }
    } catch {
        Write-ColoredOutput "ERROR: Test execution failed: $_" $Red
    }
}

# Install if requested
if ($Install) {
    Write-ColoredOutput "Installing..." $Yellow
    
    try {
        & cmake --install . --config $BuildType
        if ($LASTEXITCODE -ne 0) {
            throw "Installation failed"
        }
        Write-ColoredOutput "✓ Installation successful" $Green
    } catch {
        Write-ColoredOutput "ERROR: Installation failed: $_" $Red
        exit 1
    }
}

# Summary
Write-Host ""
Write-ColoredOutput "=======================================" $Blue
Write-ColoredOutput "           Build Summary" $Blue
Write-ColoredOutput "=======================================" $Blue
Write-ColoredOutput "Build Type: $BuildType" $Green
Write-ColoredOutput "Generator: $Generator" $Green
Write-ColoredOutput "Tests: $($Tests ? 'Enabled' : 'Disabled')" $Green
Write-ColoredOutput "Examples: $($Examples ? 'Enabled' : 'Disabled')" $Green
Write-ColoredOutput "Benchmarks: $($Benchmarks ? 'Enabled' : 'Disabled')" $Green

# Show binary locations
Write-Host ""
Write-ColoredOutput "Binary locations:" $Yellow
if (Test-Path "bin") {
    Get-ChildItem -Path "bin" -Recurse -File -Include "*.exe" | ForEach-Object {
        Write-ColoredOutput "  $($_.FullName)" $Blue
    }
}

Write-Host ""
Write-ColoredOutput "Build completed successfully!" $Green
Write-ColoredOutput "You can now run the examples or tests from the build directory." $Blue

# Return to original directory
Set-Location $ProjectRoot
