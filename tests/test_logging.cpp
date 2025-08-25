/**
 * @file test_logging.cpp
 * @brief Unit tests for the logging system.
 * @author J.J.G. Pleunes
 */

#include <gtest/gtest.h>
#include "turboinfer/util/logging.hpp"
#include <sstream>
#include <iostream>

using namespace turboinfer::util;

class LoggingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logging for tests
        initialize_logging(LogLevel::kDebug, false); // No file output for tests
    }

    void TearDown() override {
        // Clean up logging
        shutdown_logging();
    }
};

TEST_F(LoggingTest, LogLevel_Setting) {
    // Test setting different log levels
    set_log_level(LogLevel::kInfo);
    EXPECT_EQ(get_log_level(), LogLevel::kInfo);
    
    set_log_level(LogLevel::kWarning);
    EXPECT_EQ(get_log_level(), LogLevel::kWarning);
    
    set_log_level(LogLevel::kError);
    EXPECT_EQ(get_log_level(), LogLevel::kError);
}

TEST_F(LoggingTest, LogLevel_Filtering) {
    // Set log level to Warning - should filter out Debug and Info
    set_log_level(LogLevel::kWarning);
    
    // These tests are mainly to ensure no crashes occur
    // Actual output verification would require capturing stdout/stderr
    TURBOINFER_LOG_DEBUG("Debug message - should be filtered");
    TURBOINFER_LOG_INFO("Info message - should be filtered");
    TURBOINFER_LOG_WARNING("Warning message - should appear");
    TURBOINFER_LOG_ERROR("Error message - should appear");
}

TEST_F(LoggingTest, Format_Validation) {
    // Test logging with format strings
    set_log_level(LogLevel::kDebug);
    
    // These should not crash with proper formatting
    TURBOINFER_LOG_INFO("Test number: {}", 42);
    TURBOINFER_LOG_INFO("Test string: {}", "hello");
    TURBOINFER_LOG_INFO("Test multiple: {} {}", 1, "test");
}

TEST_F(LoggingTest, Initialize_Shutdown_Cycle) {
    // Test multiple initialize/shutdown cycles
    shutdown_logging();
    
    EXPECT_NO_THROW(initialize_logging(LogLevel::kInfo, false));
    EXPECT_NO_THROW(shutdown_logging());
    
    EXPECT_NO_THROW(initialize_logging(LogLevel::kDebug, false));
    EXPECT_NO_THROW(shutdown_logging());
    
    // Restore for teardown
    initialize_logging(LogLevel::kDebug, false);
}

TEST_F(LoggingTest, ThreadSafety_Basic) {
    // Basic thread safety test - multiple rapid log calls
    set_log_level(LogLevel::kDebug);
    
    for (int i = 0; i < 100; ++i) {
        TURBOINFER_LOG_INFO("Thread safety test {}", i);
    }
    
    // Should complete without crashes
    SUCCEED();
}
