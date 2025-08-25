cmake_minimum_required(VERSION 3.20)

# Find GoogleTest
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    # Download and build GoogleTest if not found
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9fd7188e9a96b3d3b4df064.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Test sources
set(TEST_SOURCES
    test_tensor.cpp
    test_tensor_engine.cpp
    test_model_loader.cpp
    test_main.cpp
)

# Create test executable
add_executable(turboinfer_tests ${TEST_SOURCES})

target_link_libraries(turboinfer_tests
    turboinfer
    gtest
    gtest_main
)

target_include_directories(turboinfer_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set target properties
set_target_properties(turboinfer_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(turboinfer_tests)

# Individual test files for easier debugging
add_executable(tensor_test test_tensor.cpp test_main.cpp)
target_link_libraries(tensor_test turboinfer gtest gtest_main)
set_target_properties(tensor_test PROPERTIES CXX_STANDARD 20)

# Install tests
install(TARGETS turboinfer_tests tensor_test
    RUNTIME DESTINATION bin/tests
)
