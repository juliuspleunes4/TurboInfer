# TurboInfer Comprehensive Test Suite
# Usage: .\scripts\run_all_tests.ps1 [options]

param(
    [string]$TestFilter = "*",
    [switch]$Verbose = $false,
    [switch]$Performance = $false,
    [switch]$Coverage = $false,
    [string]$OutputFormat = "pretty"
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
        Write-Host "❌ Build directory not found. Please build the project first." -ForegroundColor Red
        Write-Host "Run: .\scripts\dev.ps1 build" -ForegroundColor Yellow
        return $false
    }
    
    # Check if test executable exists
    if (-not (Test-Path "build/tests/turboinfer_tests.exe")) {
        Write-Host "❌ Test executable not found. Building tests..." -ForegroundColor Yellow
        & cmake --build build --target turboinfer_tests --config Debug
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Failed to build tests!" -ForegroundColor Red
            return $false
        }
    }
    
    Write-Host "✅ Prerequisites check passed" -ForegroundColor Green
    return $true
}

function Invoke-UnitTests {
    Write-Section "Running Unit Tests"
    
    $testArgs = @()
    
    if ($TestFilter -ne "*") {
        $testArgs += "--gtest_filter=$TestFilter"
    }
    
    if ($Verbose) {
        $testArgs += "--gtest_print_time=1"
        $testArgs += "--gtest_color=yes"
    }
    
    if ($OutputFormat -eq "xml") {
        $testArgs += "--gtest_output=xml:test_results.xml"
    }
    
    Write-Host "Running: build/tests/turboinfer_tests.exe $($testArgs -join ' ')" -ForegroundColor Gray
    
    & build/tests/turboinfer_tests.exe @testArgs
    $testExitCode = $LASTEXITCODE
    
    if ($testExitCode -eq 0) {
        Write-Host "✅ All unit tests passed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Some unit tests failed (exit code: $testExitCode)" -ForegroundColor Red
    }
    
    return $testExitCode
}

function Invoke-IntegrationTests {
    Write-Section "Running Integration Tests"
    
    # Test library initialization cycle
    Write-Host "Testing library initialization..." -ForegroundColor Yellow
    
    if (Test-Path "tools/test_library.py") {
        python tools/test_library.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Integration test passed" -ForegroundColor Green
        } else {
            Write-Host "❌ Integration test failed" -ForegroundColor Red
            return $LASTEXITCODE
        }
    } else {
        Write-Host "⚠️ Integration test script not found" -ForegroundColor Yellow
    }
    
    # Test example compilation and execution
    Write-Host "Testing example compilation..." -ForegroundColor Yellow
    
    & .\scripts\compile_example.ps1 readme_example
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Example compilation passed" -ForegroundColor Green
        
        # Run the example
        Write-Host "Testing example execution..." -ForegroundColor Yellow
        & .\examples\readme_example.exe
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Example execution passed" -ForegroundColor Green
        } else {
            Write-Host "❌ Example execution failed" -ForegroundColor Red
            return $LASTEXITCODE
        }
    } else {
        Write-Host "❌ Example compilation failed" -ForegroundColor Red
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
        Write-Host "✅ Performance tests completed" -ForegroundColor Green
    } else {
        Write-Host "❌ Performance tests failed" -ForegroundColor Red
    }
    
    return $perfExitCode
}

function Invoke-MemoryTests {
    Write-Section "Running Memory Tests"
    
    # Run memory-related tests
    & build/tests/turboinfer_tests.exe --gtest_filter="MemoryTest*:*Memory*" --gtest_print_time=1
    $memExitCode = $LASTEXITCODE
    
    if ($memExitCode -eq 0) {
        Write-Host "✅ Memory tests passed" -ForegroundColor Green
    } else {
        Write-Host "❌ Memory tests failed" -ForegroundColor Red
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
    
    Write-Host "✅ Test report generated: $reportFile" -ForegroundColor Green
}

function Show-TestSummary($unitResult, $integrationResult, $memoryResult, $performanceResult) {
    Write-Banner "Test Summary"
    
    $totalTests = 4
    $passedTests = 0
    
    Write-Host "Test Results:" -ForegroundColor White
    
    if ($unitResult -eq 0) {
        Write-Host "  ✅ Unit Tests: PASSED" -ForegroundColor Green
        $passedTests++
    } else {
        Write-Host "  ❌ Unit Tests: FAILED" -ForegroundColor Red
    }
    
    if ($integrationResult -eq 0) {
        Write-Host "  ✅ Integration Tests: PASSED" -ForegroundColor Green
        $passedTests++
    } else {
        Write-Host "  ❌ Integration Tests: FAILED" -ForegroundColor Red
    }
    
    if ($memoryResult -eq 0) {
        Write-Host "  ✅ Memory Tests: PASSED" -ForegroundColor Green
        $passedTests++
    } else {
        Write-Host "  ❌ Memory Tests: FAILED" -ForegroundColor Red
    }
    
    if ($Performance) {
        if ($performanceResult -eq 0) {
            Write-Host "  ✅ Performance Tests: PASSED" -ForegroundColor Green
            $passedTests++
        } else {
            Write-Host "  ❌ Performance Tests: FAILED" -ForegroundColor Red
        }
    } else {
        Write-Host "  ⏭️ Performance Tests: SKIPPED" -ForegroundColor Yellow
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

function Show-Help {
    Write-Banner "TurboInfer Test Suite Help"
    Write-Host "Usage: .\scripts\run_all_tests.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  -TestFilter <pattern>    Run only tests matching pattern (default: '*')" -ForegroundColor Green
    Write-Host "  -Verbose                 Enable verbose output" -ForegroundColor Green
    Write-Host "  -Performance             Include performance tests" -ForegroundColor Green
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
if ($args -contains "-h" -or $args -contains "--help" -or $args -contains "help") {
    Show-Help
    exit 0
}

# Main execution
Write-Banner "TurboInfer Comprehensive Test Suite"

# Check prerequisites
if (-not (Test-Prerequisites)) {
    exit 1
}

# Run test suites
$unitResult = Invoke-UnitTests
$integrationResult = Invoke-IntegrationTests
$memoryResult = Invoke-MemoryTests
$performanceResult = 0

if ($Performance) {
    $performanceResult = Invoke-PerformanceTests
}

# Generate report
New-TestReport

# Show summary and exit with appropriate code
$finalResult = Show-TestSummary $unitResult $integrationResult $memoryResult $performanceResult
exit $finalResult
