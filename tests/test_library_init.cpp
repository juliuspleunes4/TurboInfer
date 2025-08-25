/**
 * @file test_library_init.cpp
 * @brief Unit tests for library initialization and shutdown.
 * @author J.J.G. Pleunes
 */

#include <gtest/gtest.h>
#include "turboinfer/turboinfer.hpp"
#include <string>

class LibraryInitTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure clean state before each test
        if (turboinfer::is_initialized()) {
            turboinfer::shutdown();
        }
    }

    void TearDown() override {
        // Clean up after each test
        if (turboinfer::is_initialized()) {
            turboinfer::shutdown();
        }
    }
};

TEST_F(LibraryInitTest, Basic_Initialize_Shutdown) {
    // Test basic initialization
    EXPECT_FALSE(turboinfer::is_initialized());
    
    bool result = turboinfer::initialize(false); // No verbose logging
    EXPECT_TRUE(result);
    EXPECT_TRUE(turboinfer::is_initialized());
    
    // Test shutdown
    turboinfer::shutdown();
    EXPECT_FALSE(turboinfer::is_initialized());
}

TEST_F(LibraryInitTest, Initialize_With_Verbose_Logging) {
    EXPECT_FALSE(turboinfer::is_initialized());
    
    bool result = turboinfer::initialize(true); // Verbose logging
    EXPECT_TRUE(result);
    EXPECT_TRUE(turboinfer::is_initialized());
    
    turboinfer::shutdown();
    EXPECT_FALSE(turboinfer::is_initialized());
}

TEST_F(LibraryInitTest, Multiple_Initialize_Calls) {
    // First initialization should succeed
    EXPECT_TRUE(turboinfer::initialize(false));
    EXPECT_TRUE(turboinfer::is_initialized());
    
    // Second initialization should handle gracefully
    EXPECT_TRUE(turboinfer::initialize(false));
    EXPECT_TRUE(turboinfer::is_initialized());
    
    turboinfer::shutdown();
    EXPECT_FALSE(turboinfer::is_initialized());
}

TEST_F(LibraryInitTest, Shutdown_Without_Initialize) {
    // Should handle shutdown when not initialized
    EXPECT_FALSE(turboinfer::is_initialized());
    EXPECT_NO_THROW(turboinfer::shutdown());
    EXPECT_FALSE(turboinfer::is_initialized());
}

TEST_F(LibraryInitTest, Multiple_Shutdown_Calls) {
    // Initialize first
    EXPECT_TRUE(turboinfer::initialize(false));
    EXPECT_TRUE(turboinfer::is_initialized());
    
    // First shutdown
    turboinfer::shutdown();
    EXPECT_FALSE(turboinfer::is_initialized());
    
    // Second shutdown should handle gracefully
    EXPECT_NO_THROW(turboinfer::shutdown());
    EXPECT_FALSE(turboinfer::is_initialized());
}

TEST_F(LibraryInitTest, Version_Info) {
    // Test version and build info
    std::string version = turboinfer::version();
    EXPECT_FALSE(version.empty());
    EXPECT_NE(version.find("1.0.0"), std::string::npos);
    
    std::string build_info = turboinfer::build_info();
    EXPECT_FALSE(build_info.empty());
    EXPECT_NE(build_info.find("TurboInfer"), std::string::npos);
    EXPECT_NE(build_info.find("C++"), std::string::npos);
}

TEST_F(LibraryInitTest, Initialize_Shutdown_Cycle) {
    // Test multiple init/shutdown cycles
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(turboinfer::initialize(false));
        EXPECT_TRUE(turboinfer::is_initialized());
        
        // Test that functions work during initialized state
        std::string version = turboinfer::version();
        EXPECT_FALSE(version.empty());
        
        turboinfer::shutdown();
        EXPECT_FALSE(turboinfer::is_initialized());
    }
}
