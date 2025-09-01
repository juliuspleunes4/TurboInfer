@echo off
echo Running all fixed test files...

set TOTAL_TESTS=0
set TOTAL_PASSED=0
set ALL_PASSED=1

echo.
echo Running test_logging.exe...
tests\exe\test_logging.exe
if %ERRORLEVEL% neq 0 set ALL_PASSED=0

echo.
echo Running test_tensor_ops.exe...
tests\exe\test_tensor_ops.exe
if %ERRORLEVEL% neq 0 set ALL_PASSED=0

echo.
echo Running test_tensor_ops_new.exe...
if exist tests\exe\test_tensor_ops_new.exe (
    tests\exe\test_tensor_ops_new.exe
    if %ERRORLEVEL% neq 0 set ALL_PASSED=0
) else (
    echo test_tensor_ops_new.exe not found, skipping...
)

echo.
echo Running test_data_types_fixed.exe...
tests\exe\test_data_types_fixed.exe
if %ERRORLEVEL% neq 0 set ALL_PASSED=0

echo.
echo Running test_tensor_fixed.exe...
tests\exe\test_tensor_fixed.exe
if %ERRORLEVEL% neq 0 set ALL_PASSED=0

echo.
echo ===============================================
if %ALL_PASSED% equ 1 (
    echo ALL TESTS PASSED!
    exit /b 0
) else (
    echo SOME TESTS FAILED!
    exit /b 1
)
