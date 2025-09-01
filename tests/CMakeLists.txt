cmake_minimum_required(VERSION 3.20)

# Manual testing approach - no external dependencies needed

# Test sources (each is a standalone executable)
set(TEST_SOURCES
    test_tensor.cpp
    test_tensor_engine.cpp
    test_model_loader.cpp
    test_logging.cpp
    test_library_init.cpp
    test_data_types.cpp
    test_tensor_ops.cpp
    test_error_handling.cpp
    test_quantization.cpp
    test_quantization_complete.cpp
    test_beam_search_simple.cpp
    test_beam_search.cpp
    test_inference_engine.cpp
    test_phase4_complete.cpp
    test_new_tensor_ops.cpp
    test_quantization_persistence.cpp
    test_enhanced_tokenization.cpp
    test_enhanced_transformer.cpp
    test_kv_cache_incremental.cpp
    test_performance_stats.cpp
    test_incomplete_features_complete.cpp
)

# Create individual test executables (manual testing approach)
foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} turboinfer)
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(${TEST_NAME} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
    
    # Fix Windows MinGW console subsystem issue
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_options(${TEST_NAME} PRIVATE "-mconsole")
    endif()
    
    # Add to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# Create test_main as an informational executable
add_executable(test_main test_main.cpp)
set_target_properties(test_main PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Install test executables
foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    install(TARGETS ${TEST_NAME} RUNTIME DESTINATION bin/tests)
endforeach()
