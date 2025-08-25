# TurboInfer Simple Test Runner
# Usage: .\scripts\run_tests_simple.ps1

Write-Host "🚀 TurboInfer Test Runner" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host ""

# List of all test executables
$tests = @(
    "test_library_init",
    "test_tensor", 
    "test_tensor_engine",
    "test_memory",
    "test_error_handling", 
    "test_performance",
    "test_logging",
    "test_data_types",
    "test_tensor_ops",
    "test_model_loader",
    "test_quantization"
)

$totalTests = 0
$passedTests = 0
$failedTests = @()

# Check if build directory exists
if (-not (Test-Path "build")) {
    Write-Host "❌ Build directory not found. Please build the project first." -ForegroundColor Red
    Write-Host "Run: cmake --build build" -ForegroundColor Yellow
    exit 1
}

# Run each test
foreach ($testName in $tests) {
    $testPath = "build\bin\$testName.exe"
    
    # Build test if it doesn't exist
    if (-not (Test-Path $testPath)) {
        Write-Host "🔨 Building $testName..." -ForegroundColor Yellow
        Set-Location build
        cmake --build . --target $testName
        Set-Location ..
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Failed to build $testName" -ForegroundColor Red
            $failedTests += $testName
            continue
        }
    }
    
    Write-Host ""
    Write-Host "🧪 Running $testName..." -ForegroundColor Cyan
    Write-Host "-" * 50 -ForegroundColor Gray
    
    $totalTests++
    
    # Run the test
    & $testPath
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $testName PASSED" -ForegroundColor Green
        $passedTests++
    } else {
        Write-Host "❌ $testName FAILED (exit code: $LASTEXITCODE)" -ForegroundColor Red
        $failedTests += $testName
    }
}

# Summary
Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "📊 TEST SUMMARY" -ForegroundColor White
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Total tests: $totalTests" -ForegroundColor White
Write-Host "Passed: $passedTests" -ForegroundColor Green
Write-Host "Failed: $($failedTests.Count)" -ForegroundColor Red

if ($failedTests.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed tests:" -ForegroundColor Red
    foreach ($test in $failedTests) {
        Write-Host "  - $test" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "❌ SOME TESTS FAILED!" -ForegroundColor Red
    exit 1
} else {
    Write-Host ""
    Write-Host "🎉 ALL TESTS PASSED!" -ForegroundColor Green
    exit 0
}
