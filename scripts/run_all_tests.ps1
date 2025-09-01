# TurboInfer Comprehensive Test Suite (Manual Testing)
# Usage: .\scripts\run_all_tests.ps1 [options]

param(
    [string]$TestFilter = "*",
    [switch]$Verbose = $false,
    [switch]$Performance = $false,
    [string]$OutputFormat = "pretty",
    [switch]$Help = $false
)

# List of all test executables
$TEST_EXECUTABLES = @(
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

function Write-Banner($Message) {
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
}

function Write-Section($Message) {
    Write-Host ""
    Write-Host "-" * 40 -ForegroundColor Yellow
    Write-Host $Message -ForegroundColor Yellow
    Write-Host "-" * 40 -ForegroundColor Yellow
}

function Test-Prerequisites {
    Write-Section "Checking Prerequisites"
    
    # Check if build directory exists
    if (-not (Test-Path "build")) {
        Write-Host "ERROR: Build directory not found. Please build the project first." -ForegroundColor Red
        Write-Host "Run: .\scripts\dev.ps1 build" -ForegroundColor Yellow
        return $false
    }
    
    # Check if bin directory exists
    if (-not (Test-Path "build/bin")) {
        Write-Host "WARNING: Bin directory not found. Building tests..." -ForegroundColor Yellow
        Set-Location "build"
        & cmake --build . --config Debug
        Set-Location ".."
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to build tests!" -ForegroundColor Red
            return $false
        }
    }
    
    Write-Host "SUCCESS: Prerequisites check passed" -ForegroundColor Green
    return $true
}

function Invoke-AllTests {
    Write-Section "Running Individual Tests"
    
    $totalTests = 0
    $passedTests = 0
    $failedTests = @()
    
    # Filter tests if specified
    $testsToRun = $TEST_EXECUTABLES
    if ($TestFilter -ne "*") {
        $testsToRun = $TEST_EXECUTABLES | Where-Object { $_ -like "*$TestFilter*" }
    }
    
    foreach ($testName in $testsToRun) {
        $testPath = "build\bin\$testName.exe"
        
        # Build test if not exists
        if (-not (Test-Path $testPath)) {
            Write-Host "Building $testName..." -ForegroundColor Yellow
            Set-Location "build"
            & cmake --build . --target $testName --config Debug
            Set-Location ".."
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Failed to build $testName!" -ForegroundColor Red
                continue
            }
        }
        
        Write-Host ""
        Write-Host "Running $testName..." -ForegroundColor Cyan
        Write-Host "-" * 50 -ForegroundColor Gray
        
        $totalTests++
        
        # Run the test
        & $testPath
        $testExitCode = $LASTEXITCODE
        
        if ($testExitCode -eq 0) {
            Write-Host "PASSED: $testName" -ForegroundColor Green
            $passedTests++
        } else {
            Write-Host "FAILED: $testName (exit code: $testExitCode)" -ForegroundColor Red
            $failedTests += $testName
        }
    }
    
    # Summary
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host "TEST SUMMARY" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host "Total tests run: $totalTests" -ForegroundColor White
    Write-Host "Tests passed: $passedTests" -ForegroundColor Green
    Write-Host "Tests failed: $($failedTests.Count)" -ForegroundColor Red
    
    if ($failedTests.Count -gt 0) {
        Write-Host "Failed tests:" -ForegroundColor Red
        foreach ($test in $failedTests) {
            Write-Host "  - $test" -ForegroundColor Red
        }
        return 1
    } else {
        Write-Host "ALL TESTS PASSED!" -ForegroundColor Green
        return 0
    }
}

function Invoke-IntegrationTests {
    Write-Section "Running Integration Tests"
    
    # Test library initialization using C++ tests
    Write-Host "Testing library initialization..." -ForegroundColor Yellow
    
    if (Test-Path "build/bin/test_library_init.exe") {
        & "build/bin/test_library_init.exe" | Out-Host
        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: Library initialization test passed" -ForegroundColor Green
        } else {
            Write-Host "ERROR: Library initialization test failed" -ForegroundColor Red
            return $LASTEXITCODE
        }
    } else {
        Write-Host "WARNING: Library initialization test executable not found" -ForegroundColor Yellow
    }
    
    # Test example compilation and execution
    Write-Host "Testing example compilation..." -ForegroundColor Yellow
    
    & .\scripts\compile_example.ps1 readme_example
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: Example compilation passed" -ForegroundColor Green
        
        # Run the example
        Write-Host "Testing example execution..." -ForegroundColor Yellow
        & .\examples\readme_example.exe
        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: Example execution passed" -ForegroundColor Green
        } else {
            Write-Host "ERROR: Example execution failed" -ForegroundColor Red
            return $LASTEXITCODE
        }
    } else {
        Write-Host "ERROR: Example compilation failed" -ForegroundColor Red
        return $LASTEXITCODE
    }
    
    return 0
}

function Invoke-PerformanceTests {
    if (-not $Performance) {
        return 0
    }
    
    Write-Section "Running Performance Tests"
    
    # Run performance-specific tests
    & build/tests/turboinfer_tests.exe --gtest_filter="PerformanceTest*" --gtest_print_time=1
    $perfExitCode = $LASTEXITCODE
    
    if ($perfExitCode -eq 0) {
        Write-Host "SUCCESS: Performance tests completed" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Performance tests failed" -ForegroundColor Red
    }
    
    return $perfExitCode
}

function Invoke-MemoryTests {
    Write-Section "Running Memory Tests"
    
    # Run memory-related tests
    & build/tests/turboinfer_tests.exe --gtest_filter="MemoryTest*:*Memory*" --gtest_print_time=1
    $memExitCode = $LASTEXITCODE
    
    if ($memExitCode -eq 0) {
        Write-Host "SUCCESS: Memory tests passed" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Memory tests failed" -ForegroundColor Red
    }
    
    return $memExitCode
}

function New-TestReport {
    Write-Section "Generating Test Report"
    
    $reportFile = "test_report.txt"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    @"
TurboInfer Test Report
=====================
Generated: $timestamp
Filter: $TestFilter
Performance Tests: $Performance
Verbose Mode: $Verbose

Test Categories:
- Unit Tests: Core functionality testing
- Integration Tests: End-to-end workflow testing
- Memory Tests: Memory management and RAII testing
- Performance Tests: Benchmarking and performance validation
- Error Handling Tests: Exception safety and error conditions

"@ | Out-File $reportFile
    
    Write-Host "SUCCESS: Test report generated: $reportFile" -ForegroundColor Green
}

function Write-TestSummary($unitResult, $integrationResult, $memoryResult, $performanceResult) {
    Write-Banner "Test Summary"
    
    $totalTests = 4
    $passedTests = 0
    
    Write-Host "Test Results:" -ForegroundColor White
    
    if ($unitResult -eq 0) {
        Write-Host "  PASS: Unit Tests" -ForegroundColor Green
        $passedTests++
    } else {
        Write-Host "  FAIL: Unit Tests" -ForegroundColor Red
    }
    
    if ($integrationResult -eq 0) {
        Write-Host "  PASS: Integration Tests" -ForegroundColor Green
        $passedTests++
    } else {
        Write-Host "  FAIL: Integration Tests" -ForegroundColor Red
    }
    
    if ($memoryResult -eq 0) {
        Write-Host "  PASS: Memory Tests" -ForegroundColor Green
        $passedTests++
    } else {
        Write-Host "  FAIL: Memory Tests" -ForegroundColor Red
    }
    
    if ($Performance) {
        if ($performanceResult -eq 0) {
            Write-Host "  PASS: Performance Tests" -ForegroundColor Green
            $passedTests++
        } else {
            Write-Host "  FAIL: Performance Tests" -ForegroundColor Red
        }
    } else {
        Write-Host "  SKIP: Performance Tests" -ForegroundColor Yellow
        $totalTests--
    }
    
    Write-Host ""
    Write-Host "Overall Result: $passedTests/$totalTests tests passed" -ForegroundColor White
    
    if ($passedTests -eq $totalTests) {
        Write-Host "🎉 ALL TESTS PASSED!" -ForegroundColor Green
        return 0
    } else {
        Write-Host "💥 SOME TESTS FAILED!" -ForegroundColor Red
        return 1
    }
}

function Write-Help {
    Write-Banner "TurboInfer Test Suite Help"
    Write-Host "Usage: .\scripts\run_all_tests.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  -TestFilter <pattern>    Run only tests matching pattern (default: '*')" -ForegroundColor Green
    Write-Host "  -Verbose                 Enable verbose output" -ForegroundColor Green
    Write-Host "  -Performance             Include performance tests" -ForegroundColor Green
    Write-Host "  -Help                    Show this help message" -ForegroundColor Green
    Write-Host "  -Coverage                Generate coverage report (requires tools)" -ForegroundColor Green
    Write-Host "  -OutputFormat <format>   Output format: 'pretty' or 'xml'" -ForegroundColor Green
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor White
    Write-Host "  .\scripts\run_all_tests.ps1                    # Run all basic tests" -ForegroundColor Gray
    Write-Host "  .\scripts\run_all_tests.ps1 -Verbose           # Run with verbose output" -ForegroundColor Gray
    Write-Host "  .\scripts\run_all_tests.ps1 -Performance       # Include performance tests" -ForegroundColor Gray
    Write-Host "  .\scripts\run_all_tests.ps1 -TestFilter 'Tensor*'  # Run only tensor tests" -ForegroundColor Gray
}

# Check for help request
if ($Help -or $args -contains "-h" -or $args -contains "--help" -or $args -contains "help") {
    Write-Help
    exit 0
}

# Main execution
Write-Banner "TurboInfer Comprehensive Test Suite (Manual Testing)"

# Check prerequisites
if (-not (Test-Prerequisites)) {
    exit 1
}

# Run all individual tests
$result = Invoke-AllTests

# Exit with test result
exit $result
