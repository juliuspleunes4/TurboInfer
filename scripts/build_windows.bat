@echo off
REM TurboInfer Build Script for Windows
REM This script sets up the environment and builds the project

echo ===================================================
echo TurboInfer Build Script
echo ===================================================

REM Check for required tools
echo Checking for required tools...

where cmake >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: CMake not found in PATH
    echo Please make sure CMake is installed and restart PowerShell
    pause
    exit /b 1
)

where g++ >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: g++ compiler not found in PATH
    echo Please restart PowerShell to refresh environment variables
    pause
    exit /b 1
)

echo CMake version:
cmake --version

echo.
echo GCC version:
g++ --version

echo.
echo ===================================================
echo Building TurboInfer...
echo ===================================================

REM Create build directory
if not exist build mkdir build

REM Configure with CMake
echo Configuring project...
cmake -B build -S . -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=20

if %ERRORLEVEL% neq 0 (
    echo Error: CMake configuration failed
    pause
    exit /b 1
)

REM Build the project
echo Building project...
cmake --build build --config Debug --parallel

if %ERRORLEVEL% neq 0 (
    echo Error: Build failed
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Build completed successfully!
echo ===================================================

echo Running tests...
cd build
ctest --output-on-failure

echo.
echo Build and test completed!
pause
