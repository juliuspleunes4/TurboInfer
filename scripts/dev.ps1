# TurboInfer Quick Development Script
# Usage: .\scripts\dev.ps1 [action]
# Actions: build, test, clean, help

param(
    [string]$Action = "help"
)

function Write-Banner($Message) {
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
}

function Build-Project {
    Write-Banner "Building TurboInfer"
    
    $configCmd = "cmake -B build -S . -G `"MinGW Makefiles`" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=20 -DTURBOINFER_BUILD_TESTS=OFF -DTURBOINFER_BUILD_EXAMPLES=OFF -DTURBOINFER_BUILD_BENCHMARKS=OFF"
    Write-Host "Configuring: $configCmd"
    Invoke-Expression $configCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Building..." -ForegroundColor Yellow
        cmake --build build --config Debug --parallel
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Build successful!" -ForegroundColor Green
            Get-ChildItem build/lib -Name
        } else {
            Write-Host "❌ Build failed!" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ Configuration failed!" -ForegroundColor Red
    }
}

function Test-Library {
    Write-Banner "Testing TurboInfer"
    
    if (-not (Test-Path "build/lib/libturboinfer.a")) {
        Write-Host "❌ Library not found. Building first..." -ForegroundColor Yellow
        Build-Project
        if ($LASTEXITCODE -ne 0) { return }
    }
    
    # Run library initialization test
    if (Test-Path "build/bin/test_library_init.exe") {
        Write-Host "Running library initialization test..." -ForegroundColor Yellow
        & "build/bin/test_library_init.exe"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Library test passed" -ForegroundColor Green
        } else {
            Write-Host "❌ Library test failed" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ Library test executable not found" -ForegroundColor Red
    }
}

function Clean-Project {
    Write-Banner "Cleaning TurboInfer"
    
    if (Test-Path "build") {
        Remove-Item "build" -Recurse -Force
        Write-Host "✅ Removed build directory" -ForegroundColor Green
    }
    
    # Remove any temporary files
    Get-ChildItem -Name "test_*" | ForEach-Object {
        Remove-Item $_ -Force
        Write-Host "✅ Removed $_" -ForegroundColor Green
    }
    
    Write-Host "✅ Project cleaned!" -ForegroundColor Green
}

function Write-Help {
    Write-Banner "TurboInfer Development Helper"
    Write-Host "Usage: .\scripts\dev.ps1 [action]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Actions:" -ForegroundColor White
    Write-Host "  build  - Configure and build the project" -ForegroundColor Green
    Write-Host "  test   - Run library tests" -ForegroundColor Green
    Write-Host "  clean  - Clean build artifacts" -ForegroundColor Green
    Write-Host "  help   - Show this help message" -ForegroundColor Green
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor White
    Write-Host "  .\scripts\dev.ps1 build" -ForegroundColor Gray
    Write-Host "  .\scripts\dev.ps1 test" -ForegroundColor Gray
    Write-Host "  .\scripts\dev.ps1 clean" -ForegroundColor Gray
}

# Main execution
switch ($Action.ToLower()) {
    "build" { Build-Project }
    "test" { Test-Library }
    "clean" { Clean-Project }
    "help" { Write-Help }
    default { 
        Write-Host "❌ Unknown action: $Action" -ForegroundColor Red
        Write-Help 
    }
}
