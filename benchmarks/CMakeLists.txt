# TurboInfer Benchmarks CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Create comprehensive benchmark executable
add_executable(benchmark_inference
    benchmark_inference.cpp
)

# Create simple benchmark executable for current capabilities
add_executable(benchmark_simple
    benchmark_simple.cpp
)

# Link against TurboInfer library
target_link_libraries(benchmark_inference
    PRIVATE
        turboinfer
)

target_link_libraries(benchmark_simple
    PRIVATE
        turboinfer
)

# Set C++20 standard
target_compile_features(benchmark_inference PRIVATE cxx_std_20)
target_compile_features(benchmark_simple PRIVATE cxx_std_20)

# Add compiler-specific optimizations for benchmarks
if(MSVC)
    target_compile_options(benchmark_inference PRIVATE /O2 /DNDEBUG)
    target_compile_options(benchmark_simple PRIVATE /O2 /DNDEBUG)
else()
    target_compile_options(benchmark_inference PRIVATE -O3 -DNDEBUG)
    target_compile_options(benchmark_simple PRIVATE -O3 -DNDEBUG)
endif()

# Install benchmark executables
install(TARGETS benchmark_inference benchmark_simple
    RUNTIME DESTINATION bin
)

# Add custom targets to run benchmarks
add_custom_target(run_benchmarks
    COMMAND benchmark_inference
    DEPENDS benchmark_inference
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running TurboInfer comprehensive inference benchmarks..."
)

add_custom_target(run_simple_benchmarks
    COMMAND benchmark_simple
    DEPENDS benchmark_simple
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running TurboInfer simple benchmarks..."
)
