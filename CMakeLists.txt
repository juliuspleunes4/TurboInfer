cmake_minimum_required(VERSION 3.20)
project(TurboInfer VERSION 1.0.0 LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include custom CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(TURBOINFER_BUILD_TESTS "Build unit tests" ON)
option(TURBOINFER_BUILD_EXAMPLES "Build examples" ON)
option(TURBOINFER_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TURBOINFER_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(TURBOINFER_ENABLE_OPENMP "Enable OpenMP support" ON)

# Find dependencies
find_package(Threads REQUIRED)

# OpenMP support
if(TURBOINFER_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# Eigen3 for linear algebra
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core library
file(GLOB_RECURSE TURBOINFER_SOURCES
    "src/core/*.cpp"
    "src/model/*.cpp"
    "src/optimize/*.cpp"
    "src/util/*.cpp"
)

file(GLOB_RECURSE TURBOINFER_HEADERS
    "include/turboinfer/*.hpp"
)

# Create the main library
add_library(turboinfer ${TURBOINFER_SOURCES} ${TURBOINFER_HEADERS})

target_include_directories(turboinfer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(turboinfer PUBLIC
    Threads::Threads
    Eigen3::Eigen
)

if(TURBOINFER_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(turboinfer PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(turboinfer PUBLIC TURBOINFER_OPENMP_ENABLED)
endif()

if(TURBOINFER_ENABLE_SIMD)
    target_compile_definitions(turboinfer PUBLIC TURBOINFER_SIMD_ENABLED)
    if(MSVC)
        target_compile_options(turboinfer PRIVATE /arch:AVX2)
    else()
        target_compile_options(turboinfer PRIVATE -mavx2 -mfma)
    endif()
endif()

# Set target properties
set_target_properties(turboinfer PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Tests
if(TURBOINFER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(TURBOINFER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Benchmarks
if(TURBOINFER_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation
install(TARGETS turboinfer
    EXPORT TurboInferTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/turboinfer
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Export configuration
install(EXPORT TurboInferTargets
    FILE TurboInferTargets.cmake
    NAMESPACE TurboInfer::
    DESTINATION lib/cmake/TurboInfer
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    TurboInferConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TurboInferConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TurboInferConfig.cmake
    INSTALL_DESTINATION lib/cmake/TurboInfer
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TurboInferConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TurboInferConfigVersion.cmake
    DESTINATION lib/cmake/TurboInfer
)
