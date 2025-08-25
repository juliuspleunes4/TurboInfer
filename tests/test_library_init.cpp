/**
 * @file test_library_init.cpp
 * @brief Unit tests for library initialization and shutdown.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <cassert>
#include <string>

// Test result tracking
int tests_run = 0;
int tests_passed = 0;

#define ASSERT_TRUE(condition) \
    do { \
        tests_run++; \
        if (condition) { \
            tests_passed++; \
            std::cout << "✅ PASS: " << #condition << std::endl; \
        } else { \
            std::cout << "❌ FAIL: " << #condition << std::endl; \
        } \
    } while(0)

#define ASSERT_FALSE(condition) ASSERT_TRUE(!(condition))
#define ASSERT_NO_THROW(statement) \
    do { \
        tests_run++; \
        try { \
            statement; \
            tests_passed++; \
            std::cout << "✅ PASS: " << #statement << " (no exception)" << std::endl; \
        } catch (...) { \
            std::cout << "❌ FAIL: " << #statement << " (unexpected exception)" << std::endl; \
        } \
    } while(0)

void setup_test() {
    // Ensure clean state before each test
    if (turboinfer::is_initialized()) {
        turboinfer::shutdown();
    }
}

void teardown_test() {
    // Clean up after each test
    if (turboinfer::is_initialized()) {
        turboinfer::shutdown();
    }
}

void test_basic_initialize_shutdown() {
    std::cout << "\n--- Test: Basic Initialize Shutdown ---" << std::endl;
    setup_test();
    
    // Test basic initialization
    ASSERT_FALSE(turboinfer::is_initialized());
    
    bool result = turboinfer::initialize(false); // No verbose logging
    ASSERT_TRUE(result);
    ASSERT_TRUE(turboinfer::is_initialized());
    
    // Test shutdown
    turboinfer::shutdown();
    ASSERT_FALSE(turboinfer::is_initialized());
    
    teardown_test();
}

void test_initialize_with_verbose_logging() {
    std::cout << "\n--- Test: Initialize With Verbose Logging ---" << std::endl;
    setup_test();
    
    ASSERT_FALSE(turboinfer::is_initialized());
    
    bool result = turboinfer::initialize(true); // Verbose logging
    ASSERT_TRUE(result);
    ASSERT_TRUE(turboinfer::is_initialized());
    
    turboinfer::shutdown();
    ASSERT_FALSE(turboinfer::is_initialized());
    
    teardown_test();
}

void test_multiple_initialize_calls() {
    std::cout << "\n--- Test: Multiple Initialize Calls ---" << std::endl;
    setup_test();
    
    // First initialization should succeed
    ASSERT_TRUE(turboinfer::initialize(false));
    ASSERT_TRUE(turboinfer::is_initialized());
    
    // Second initialization should handle gracefully
    ASSERT_TRUE(turboinfer::initialize(false));
    ASSERT_TRUE(turboinfer::is_initialized());
    
    turboinfer::shutdown();
    ASSERT_FALSE(turboinfer::is_initialized());
    
    teardown_test();
}

void test_shutdown_without_initialize() {
    std::cout << "\n--- Test: Shutdown Without Initialize ---" << std::endl;
    setup_test();
    
    // Should handle shutdown when not initialized
    ASSERT_FALSE(turboinfer::is_initialized());
    ASSERT_NO_THROW(turboinfer::shutdown());
    ASSERT_FALSE(turboinfer::is_initialized());
    
    teardown_test();
}

void test_multiple_shutdown_calls() {
    std::cout << "\n--- Test: Multiple Shutdown Calls ---" << std::endl;
    setup_test();
    
    // Initialize first
    ASSERT_TRUE(turboinfer::initialize(false));
    ASSERT_TRUE(turboinfer::is_initialized());
    
    // First shutdown
    turboinfer::shutdown();
    ASSERT_FALSE(turboinfer::is_initialized());
    
    // Second shutdown should handle gracefully
    ASSERT_NO_THROW(turboinfer::shutdown());
    ASSERT_FALSE(turboinfer::is_initialized());
    
    teardown_test();
}

void test_version_info() {
    std::cout << "\n--- Test: Version Info ---" << std::endl;
    setup_test();
    
    // Test version and build info
    std::string version = turboinfer::version();
    ASSERT_FALSE(version.empty());
    ASSERT_TRUE(version.find("1.0.0") != std::string::npos);
    
    std::string build_info = turboinfer::build_info();
    ASSERT_FALSE(build_info.empty());
    ASSERT_TRUE(build_info.find("TurboInfer") != std::string::npos);
    ASSERT_TRUE(build_info.find("C++") != std::string::npos);
    
    teardown_test();
}

void test_initialize_shutdown_cycle() {
    std::cout << "\n--- Test: Initialize Shutdown Cycle ---" << std::endl;
    setup_test();
    
    // Test multiple init/shutdown cycles
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(turboinfer::initialize(false));
        ASSERT_TRUE(turboinfer::is_initialized());
        
        // Test that functions work during initialized state
        std::string version = turboinfer::version();
        ASSERT_FALSE(version.empty());
        
        turboinfer::shutdown();
        ASSERT_FALSE(turboinfer::is_initialized());
    }
    
    teardown_test();
}

int main() {
    std::cout << "🚀 Starting TurboInfer Library Initialization Tests..." << std::endl;
    
    test_basic_initialize_shutdown();
    test_initialize_with_verbose_logging();
    test_multiple_initialize_calls();
    test_shutdown_without_initialize();
    test_multiple_shutdown_calls();
    test_version_info();
    test_initialize_shutdown_cycle();
    
    std::cout << "\n📊 Test Results:" << std::endl;
    std::cout << "Tests run: " << tests_run << std::endl;
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << (tests_run - tests_passed) << std::endl;
    
    if (tests_passed == tests_run) {
        std::cout << "🎉 ALL TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}
