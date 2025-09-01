#!/usr/bin/env powershell
# Script to run all the fixed test files

Write-Host "🧪 Running all fixed test files..." -ForegroundColor Cyan

$testDir = "tests\exe"
$tests = @(
    "test_logging.exe",
    "test_tensor_ops.exe", 
    "test_tensor_ops_new.exe",
    "test_data_types_fixed.exe",
    "test_tensor_fixed.exe"
)

$totalTests = 0
$totalPassed = 0
$allPassed = $true

foreach ($test in $tests) {
    $testPath = Join-Path $testDir $test
    if (Test-Path $testPath) {
        Write-Host "`n🔍 Running $test..." -ForegroundColor Yellow
        
        # Run the test and capture output
        $output = & $testPath 2>&1
        $exitCode = $LASTEXITCODE
        
        # Show the output
        Write-Host $output
        
        if ($exitCode -eq 0) {
            Write-Host "✅ $test PASSED" -ForegroundColor Green
        } else {
            Write-Host "❌ $test FAILED" -ForegroundColor Red
            $allPassed = $false
        }
        
        # Extract test counts from output
        $testRunMatch = [regex]::Match($output, "Tests run: (\d+)")
        $testPassedMatch = [regex]::Match($output, "Tests passed: (\d+)")
        
        if ($testRunMatch.Success -and $testPassedMatch.Success) {
            $totalTests += [int]$testRunMatch.Groups[1].Value
            $totalPassed += [int]$testPassedMatch.Groups[1].Value
        }
    } else {
        Write-Host "⚠️  $test not found, skipping..." -ForegroundColor Yellow
    }
}

Write-Host "`n📊 Summary:" -ForegroundColor Cyan
Write-Host "Total Tests Run: $totalTests" -ForegroundColor White
Write-Host "Total Tests Passed: $totalPassed" -ForegroundColor Green
Write-Host "Total Tests Failed: $($totalTests - $totalPassed)" -ForegroundColor Red

if ($allPassed) {
    Write-Host "`n🎉 ALL TESTS PASSED!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n💥 SOME TESTS FAILED!" -ForegroundColor Red
    exit 1
}
