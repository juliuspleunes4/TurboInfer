@echo off
echo =====================================
echo   TurboInfer Test Runner
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
echo [1/11] Running test_library_init...
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

REM Test tensor
echo [2/11] Running test_tensor...
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

REM Test memory
echo [3/11] Running test_memory...
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
echo [4/11] Running test_error_handling...
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

REM Summary
echo =====================================
echo   TEST SUMMARY
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
