@echo off
echo =======================================
echo      TurboInfer Build Script
echo =======================================
echo.

REM Check if CMake is available
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: CMake not found. Please install CMake and add it to PATH.
    echo.
    echo You can download CMake from: https://cmake.org/download/
    echo.
    pause
    exit /b 1
)

REM Check if we have a C++ compiler
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: MSVC compiler not found. Please install Visual Studio 2019+ or Build Tools.
    echo.
    echo You can download Visual Studio from: https://visualstudio.microsoft.com/
    echo.
    pause
    exit /b 1
)

REM Set build type (default to Release)
set BUILD_TYPE=Release
if not "%1"=="" set BUILD_TYPE=%1

echo Build Type: %BUILD_TYPE%
echo.

REM Create build directory
if not exist build mkdir build
cd build

echo Configuring project...
cmake .. -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DTURBOINFER_BUILD_TESTS=ON -DTURBOINFER_BUILD_EXAMPLES=ON
if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed.
    pause
    exit /b 1
)

echo.
echo Building project...
cmake --build . --config %BUILD_TYPE%
if %errorlevel% neq 0 (
    echo ERROR: Build failed.
    pause
    exit /b 1
)

echo.
echo Running tests...
ctest -C %BUILD_TYPE% --output-on-failure
if %errorlevel% neq 0 (
    echo WARNING: Some tests failed.
) else (
    echo All tests passed!
)

echo.
echo Build completed successfully!
echo Binary location: build\bin\%BUILD_TYPE%\
echo.
pause
