# TurboInfer Quick Test Categories
# Usage: .\scripts\test_category.ps1 <category>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("tensor", "memory", "logging", "init", "types", "ops", "errors", "performance", "quantization", "all")]
    [string]$Category
)

function Write-Banner($Message) {
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
}

function Invoke-TestCategory($testFilter, $description) {
    Write-Banner "Running $description"
    
    if (-not (Test-Path "build/tests/turboinfer_tests.exe")) {
        Write-Host "❌ Test executable not found. Building..." -ForegroundColor Red
        & cmake --build build --target turboinfer_tests --config Debug
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Build failed!" -ForegroundColor Red
            return $LASTEXITCODE
        }
    }
    
    Write-Host "Filter: $testFilter" -ForegroundColor Yellow
    & build/tests/turboinfer_tests.exe --gtest_filter="$testFilter" --gtest_color=yes --gtest_print_time=1
    
    return $LASTEXITCODE
}

# Define test categories
$categories = @{
    "tensor" = @{
        "filter" = "TensorTest*:TensorOpsTest*"
        "description" = "Tensor System Tests"
    }
    "memory" = @{
        "filter" = "MemoryTest*"
        "description" = "Memory Management Tests"
    }
    "logging" = @{
        "filter" = "LoggingTest*"
        "description" = "Logging System Tests"
    }
    "init" = @{
        "filter" = "LibraryInitTest*"
        "description" = "Library Initialization Tests"
    }
    "types" = @{
        "filter" = "DataTypeTest*"
        "description" = "Data Type System Tests"
    }
    "ops" = @{
        "filter" = "*Ops*:TensorOpsTest*"
        "description" = "Operations Tests"
    }
    "errors" = @{
        "filter" = "ErrorHandlingTest*"
        "description" = "Error Handling Tests"
    }
    "performance" = @{
        "filter" = "PerformanceTest*"
        "description" = "Performance Tests"
    }
    "quantization" = @{
        "filter" = "QuantizationTest*"
        "description" = "Quantization Tests"
    }
    "all" = @{
        "filter" = "*"
        "description" = "All Tests"
    }
}

if ($categories.ContainsKey($Category)) {
    $testInfo = $categories[$Category]
    $result = Invoke-TestCategory $testInfo.filter $testInfo.description
    
    if ($result -eq 0) {
        Write-Host ""
        Write-Host "✅ $($testInfo.description) - ALL PASSED!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "❌ $($testInfo.description) - SOME FAILED!" -ForegroundColor Red
    }
    
    exit $result
} else {
    Write-Host "❌ Unknown category: $Category" -ForegroundColor Red
    Write-Host "Available categories: tensor, memory, logging, init, types, ops, errors, performance, quantization, all" -ForegroundColor Yellow
    exit 1
}
