/**
 * @file test_model_loader.cpp
 * @brief Unit tests for the ModelLoader class.
 * @author J.J.G. Pleunes
 */

#include <gtest/gtest.h>
#include "turboinfer/model/model_loader.hpp"

using namespace turboinfer::model;

class ModelLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for model loader tests
    }
};

TEST_F(ModelLoaderTest, FormatDetection) {
    // Test format detection logic
    EXPECT_STREQ(format_to_string(ModelFormat::kGGUF), "GGUF");
    EXPECT_STREQ(format_to_extension(ModelFormat::kGGUF), ".gguf");
}

// Additional tests will be implemented as the ModelLoader is developed
