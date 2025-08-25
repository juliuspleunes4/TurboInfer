@echo off
echo =====================================
echo   TurboInfer Comprehensive Test Runner
echo =====================================
echo.

set TOTAL_TESTS=0
set PASSED_TESTS=0
set FAILED_TESTS=0

if not exist "build\bin" (
    echo Error: Build directory not found. Please build the project first.
    echo Run: cmake --build build
    exit /b 1
)

echo Running all tests...
echo.

REM Test library initialization
echo [1/12] Running test_library_init...
build\bin\test_library_init.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_library_init PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_library_init FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Test tensor operations
echo [2/12] Running test_tensor...
build\bin\test_tensor.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_tensor PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_tensor FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Test memory management
echo [3/12] Running test_memory...
build\bin\test_memory.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_memory PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_memory FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Test error handling
echo [4/12] Running test_error_handling...
build\bin\test_error_handling.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_error_handling PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_error_handling FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Test tensor engine
echo [5/12] Running test_tensor_engine...
build\bin\test_tensor_engine.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_tensor_engine PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_tensor_engine FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Test model loader
echo [6/12] Running test_model_loader...
build\bin\test_model_loader.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_model_loader PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_model_loader FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Test logging
echo [7/12] Running test_logging...
build\bin\test_logging.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_logging PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_logging FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Test data types
echo [8/12] Running test_data_types...
build\bin\test_data_types.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_data_types PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_data_types FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Test tensor operations
echo [9/12] Running test_tensor_ops...
build\bin\test_tensor_ops.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_tensor_ops PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_tensor_ops FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Test performance
echo [10/12] Running test_performance...
build\bin\test_performance.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_performance PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_performance FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Test quantization
echo [11/12] Running test_quantization...
build\bin\test_quantization.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_quantization PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_quantization FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Test main (informational)
echo [12/12] Running test_main...
build\bin\test_main.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ test_main PASSED
    set /a PASSED_TESTS+=1
) else (
    echo ✗ test_main FAILED
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1
echo.

REM Summary
echo =====================================
echo   COMPREHENSIVE TEST SUMMARY
echo =====================================
echo Total tests: %TOTAL_TESTS%
echo Passed: %PASSED_TESTS%
echo Failed: %FAILED_TESTS%
echo.

if %FAILED_TESTS% EQU 0 (
    echo All tests PASSED! ✓
    exit /b 0
) else (
    echo Some tests FAILED! ✗
    exit /b 1
)
