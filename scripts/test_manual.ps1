# TurboInfer Manual Test Runner
# Usage: .\scripts\test_manual.ps1

param(
    [string]$TestType = "basic"
)

function Write-Banner($Message) {
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
}

function Test-TensorBasics {
    Write-Banner "Testing Tensor Basics"
    
    $testCode = @"
#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing TurboInfer Tensor System..." << std::endl;
    
    if (!turboinfer::initialize(false)) {
        std::cerr << "Failed to initialize TurboInfer" << std::endl;
        return 1;
    }
    
    try {
        // Test 1: Basic tensor creation
        turboinfer::core::TensorShape shape({3, 4});
        turboinfer::core::Tensor tensor(shape, turboinfer::core::DataType::kFloat32);
        
        assert(tensor.shape().ndim() == 2);
        assert(tensor.shape().size(0) == 3);
        assert(tensor.shape().size(1) == 4);
        assert(tensor.shape().total_size() == 12);
        assert(tensor.dtype() == turboinfer::core::DataType::kFloat32);
        assert(tensor.byte_size() == 12 * 4); // 12 floats * 4 bytes
        
        std::cout << "✅ Test 1 (Basic Creation): PASSED" << std::endl;
        
        // Test 2: Tensor slicing
        std::vector<std::size_t> start = {0, 0};
        std::vector<std::size_t> end = {2, 3};
        auto sliced = tensor.slice(start, end);
        
        assert(sliced.shape().ndim() == 2);
        assert(sliced.shape().size(0) == 2);
        assert(sliced.shape().size(1) == 3);
        assert(sliced.shape().total_size() == 6);
        
        std::cout << "✅ Test 2 (Slicing): PASSED" << std::endl;
        
        // Test 3: Copy constructor
        turboinfer::core::Tensor copy(tensor);
        assert(copy.shape() == tensor.shape());
        assert(copy.dtype() == tensor.dtype());
        assert(copy.byte_size() == tensor.byte_size());
        assert(copy.data() != tensor.data()); // Different memory
        
        std::cout << "✅ Test 3 (Copy Constructor): PASSED" << std::endl;
        
        // Test 4: Different data types
        turboinfer::core::Tensor tensor_f16(shape, turboinfer::core::DataType::kFloat16);
        turboinfer::core::Tensor tensor_i32(shape, turboinfer::core::DataType::kInt32);
        turboinfer::core::Tensor tensor_i8(shape, turboinfer::core::DataType::kInt8);
        
        assert(tensor_f16.byte_size() == 12 * 2); // 12 * 2 bytes
        assert(tensor_i32.byte_size() == 12 * 4); // 12 * 4 bytes
        assert(tensor_i8.byte_size() == 12 * 1);  // 12 * 1 bytes
        
        std::cout << "✅ Test 4 (Data Types): PASSED" << std::endl;
        
        // Test 5: Large tensor
        turboinfer::core::TensorShape large_shape({100, 200});
        turboinfer::core::Tensor large_tensor(large_shape, turboinfer::core::DataType::kFloat32);
        assert(large_tensor.shape().total_size() == 20000);
        assert(large_tensor.byte_size() == 20000 * 4);
        
        std::cout << "✅ Test 5 (Large Tensor): PASSED" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        turboinfer::shutdown();
        return 1;
    }
    
    turboinfer::shutdown();
    std::cout << "🎉 All tensor tests PASSED!" << std::endl;
    return 0;
}
"@
    
    # Write test code to file
    $testCode | Out-File -FilePath "test_tensor_manual.cpp" -Encoding UTF8
    
    # Compile and run
    Write-Host "Compiling tensor test..." -ForegroundColor Yellow
    & g++ -std=c++20 -I include test_tensor_manual.cpp build/lib/libturboinfer.a -o test_tensor_manual.exe
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Running tensor test..." -ForegroundColor Yellow
        & .\test_tensor_manual.exe
        $result = $LASTEXITCODE
        
        # Cleanup
        Remove-Item "test_tensor_manual.cpp" -Force -ErrorAction SilentlyContinue
        Remove-Item "test_tensor_manual.exe" -Force -ErrorAction SilentlyContinue
        
        return $result
    } else {
        Write-Host "❌ Compilation failed!" -ForegroundColor Red
        return 1
    }
}

function Test-ErrorHandling {
    Write-Banner "Testing Error Handling"
    
    $testCode = @"
#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    std::cout << "Testing TurboInfer Error Handling..." << std::endl;
    
    if (!turboinfer::initialize(false)) {
        std::cerr << "Failed to initialize TurboInfer" << std::endl;
        return 1;
    }
    
    int passed_tests = 0;
    int total_tests = 0;
    
    // Test 1: Invalid tensor shape
    total_tests++;
    try {
        std::vector<size_t> invalid_dims = {0, 5}; // Zero dimension
        turboinfer::core::TensorShape invalid_shape(invalid_dims);
        std::cout << "❌ Test 1 (Invalid Shape): Should have thrown exception" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✅ Test 1 (Invalid Shape): PASSED - " << e.what() << std::endl;
        passed_tests++;
    } catch (...) {
        std::cout << "❌ Test 1 (Invalid Shape): Wrong exception type" << std::endl;
    }
    
    // Test 2: Out of bounds slice
    total_tests++;
    try {
        turboinfer::core::TensorShape shape({5, 6});
        turboinfer::core::Tensor tensor(shape, turboinfer::core::DataType::kFloat32);
        
        std::vector<std::size_t> start = {6, 0}; // Start beyond bounds
        std::vector<std::size_t> end = {5, 6};
        auto sliced = tensor.slice(start, end);
        std::cout << "❌ Test 2 (Out of Bounds): Should have thrown exception" << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "✅ Test 2 (Out of Bounds): PASSED - " << e.what() << std::endl;
        passed_tests++;
    } catch (...) {
        std::cout << "❌ Test 2 (Out of Bounds): Wrong exception type" << std::endl;
    }
    
    // Test 3: Invalid slice dimensions
    total_tests++;
    try {
        turboinfer::core::TensorShape shape({5, 6});
        turboinfer::core::Tensor tensor(shape, turboinfer::core::DataType::kFloat32);
        
        std::vector<std::size_t> start = {3, 4};
        std::vector<std::size_t> end = {2, 3}; // Start >= end
        auto sliced = tensor.slice(start, end);
        std::cout << "❌ Test 3 (Invalid Slice): Should have thrown exception" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✅ Test 3 (Invalid Slice): PASSED - " << e.what() << std::endl;
        passed_tests++;
    } catch (...) {
        std::cout << "❌ Test 3 (Invalid Slice): Wrong exception type" << std::endl;
    }
    
    turboinfer::shutdown();
    
    std::cout << std::endl;
    std::cout << "Error handling tests: " << passed_tests << "/" << total_tests << " passed" << std::endl;
    
    if (passed_tests == total_tests) {
        std::cout << "🎉 All error handling tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some error handling tests FAILED!" << std::endl;
        return 1;
    }
}
"@
    
    # Write test code to file
    $testCode | Out-File -FilePath "test_error_manual.cpp" -Encoding UTF8
    
    # Compile and run
    Write-Host "Compiling error handling test..." -ForegroundColor Yellow
    & g++ -std=c++20 -I include test_error_manual.cpp build/lib/libturboinfer.a -o test_error_manual.exe
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Running error handling test..." -ForegroundColor Yellow
        & .\test_error_manual.exe
        $result = $LASTEXITCODE
        
        # Cleanup
        Remove-Item "test_error_manual.cpp" -Force -ErrorAction SilentlyContinue
        Remove-Item "test_error_manual.exe" -Force -ErrorAction SilentlyContinue
        
        return $result
    } else {
        Write-Host "❌ Compilation failed!" -ForegroundColor Red
        return 1
    }
}

function Test-Performance {
    Write-Banner "Testing Performance"
    
    $testCode = @"
#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <chrono>
#include <vector>

int main() {
    std::cout << "Testing TurboInfer Performance..." << std::endl;
    
    if (!turboinfer::initialize(false)) {
        std::cerr << "Failed to initialize TurboInfer" << std::endl;
        return 1;
    }
    
    // Test 1: Tensor creation performance
    const int num_iterations = 1000;
    std::vector<double> times;
    
    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        turboinfer::core::TensorShape shape({100, 100});
        turboinfer::core::Tensor tensor(shape, turboinfer::core::DataType::kFloat32);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0); // Convert to milliseconds
    }
    
    double total_time = 0.0;
    for (double t : times) {
        total_time += t;
    }
    double avg_time = total_time / num_iterations;
    
    std::cout << "Tensor creation (100x100): " << avg_time << " ms average" << std::endl;
    
    if (avg_time < 1.0) {
        std::cout << "✅ Creation Performance: PASSED (< 1ms)" << std::endl;
    } else {
        std::cout << "⚠️ Creation Performance: SLOW (>= 1ms)" << std::endl;
    }
    
    // Test 2: Large tensor performance
    auto start = std::chrono::high_resolution_clock::now();
    
    turboinfer::core::TensorShape large_shape({1000, 1000});
    turboinfer::core::Tensor large_tensor(large_shape, turboinfer::core::DataType::kFloat32);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Large tensor (1000x1000): " << duration.count() << " ms" << std::endl;
    
    if (duration.count() < 100) {
        std::cout << "✅ Large Tensor Performance: PASSED (< 100ms)" << std::endl;
    } else {
        std::cout << "⚠️ Large Tensor Performance: SLOW (>= 100ms)" << std::endl;
    }
    
    // Test 3: Slicing performance
    const int slice_iterations = 10000;
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < slice_iterations; ++i) {
        std::vector<std::size_t> slice_start = {100, 100};
        std::vector<std::size_t> slice_end = {900, 900};
        auto sliced = large_tensor.slice(slice_start, slice_end);
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double slice_avg = (duration.count() / 1000.0) / slice_iterations; // ms per slice
    
    std::cout << "Slicing (10k iterations): " << slice_avg << " ms average per slice" << std::endl;
    
    if (slice_avg < 0.01) {
        std::cout << "✅ Slicing Performance: PASSED (< 0.01ms)" << std::endl;
    } else {
        std::cout << "⚠️ Slicing Performance: SLOW (>= 0.01ms)" << std::endl;
    }
    
    turboinfer::shutdown();
    std::cout << "🎯 Performance tests completed!" << std::endl;
    return 0;
}
"@
    
    # Write test code to file
    $testCode | Out-File -FilePath "test_perf_manual.cpp" -Encoding UTF8
    
    # Compile and run
    Write-Host "Compiling performance test..." -ForegroundColor Yellow
    & g++ -std=c++20 -I include test_perf_manual.cpp build/lib/libturboinfer.a -o test_perf_manual.exe
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Running performance test..." -ForegroundColor Yellow
        & .\test_perf_manual.exe
        $result = $LASTEXITCODE
        
        # Cleanup
        Remove-Item "test_perf_manual.cpp" -Force -ErrorAction SilentlyContinue
        Remove-Item "test_perf_manual.exe" -Force -ErrorAction SilentlyContinue
        
        return $result
    } else {
        Write-Host "❌ Compilation failed!" -ForegroundColor Red
        return 1
    }
}

# Main execution
Write-Banner "TurboInfer Manual Test Suite"

$passed = 0
$total = 0

# Check prerequisites
if (-not (Test-Path "build/lib/libturboinfer.a")) {
    Write-Host "❌ Library not found. Please build first with: .\scripts\dev.ps1 build" -ForegroundColor Red
    exit 1
}

switch ($TestType.ToLower()) {
    "basic" {
        $total++
        if ((Test-TensorBasics) -eq 0) { $passed++ }
    }
    "errors" {
        $total++
        if ((Test-ErrorHandling) -eq 0) { $passed++ }
    }
    "performance" {
        $total++
        if ((Test-Performance) -eq 0) { $passed++ }
    }
    "all" {
        $total = 3
        if ((Test-TensorBasics) -eq 0) { $passed++ }
        if ((Test-ErrorHandling) -eq 0) { $passed++ }
        if ((Test-Performance) -eq 0) { $passed++ }
    }
    default {
        Write-Host "❌ Unknown test type: $TestType" -ForegroundColor Red
        Write-Host "Available types: basic, errors, performance, all" -ForegroundColor Yellow
        exit 1
    }
}

Write-Banner "Test Summary"
Write-Host "Tests passed: $passed/$total" -ForegroundColor White

if ($passed -eq $total) {
    Write-Host "🎉 ALL TESTS PASSED!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ SOME TESTS FAILED!" -ForegroundColor Red
    exit 1
}
