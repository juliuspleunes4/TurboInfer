/**
 * @file test_model_loader.cpp
 * @brief Manual unit tests for model loading functionality.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <filesystem>

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
#define ASSERT_EQ(expected, actual) ASSERT_TRUE((expected) == (actual))
#define ASSERT_NE(expected, actual) ASSERT_TRUE((expected) != (actual))
#define ASSERT_GT(val1, val2) ASSERT_TRUE((val1) > (val2))
#define ASSERT_LT(val1, val2) ASSERT_TRUE((val1) < (val2))
#define ASSERT_GE(val1, val2) ASSERT_TRUE((val1) >= (val2))
#define ASSERT_LE(val1, val2) ASSERT_TRUE((val1) <= (val2))

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

#define ASSERT_THROW(statement, exception_type) \
    do { \
        tests_run++; \
        try { \
            statement; \
            std::cout << "❌ FAIL: " << #statement << " (expected exception)" << std::endl; \
        } catch (const exception_type&) { \
            tests_passed++; \
            std::cout << "✅ PASS: " << #statement << " (expected exception caught)" << std::endl; \
        } catch (...) { \
            std::cout << "❌ FAIL: " << #statement << " (wrong exception type)" << std::endl; \
        } \
    } while(0)

std::filesystem::path test_dir;

void setup_test() {
    // Create test directory
    test_dir = std::filesystem::temp_directory_path() / "turboinfer_test";
    std::filesystem::create_directories(test_dir);
}

void teardown_test() {
    // Clean up test files
    if (std::filesystem::exists(test_dir)) {
        std::filesystem::remove_all(test_dir);
    }
}

void test_model_metadata() {
    std::cout << "\n--- Test: Model Metadata Structure ---" << std::endl;
    setup_test();
    
    // Test ModelMetadata structure
    turboinfer::model::ModelMetadata metadata;
    metadata.name = "test_model";
    metadata.architecture = "llama";
    metadata.version = "1.0";
    metadata.vocab_size = 32000;
    metadata.hidden_size = 4096;
    metadata.num_layers = 32;
    metadata.num_heads = 32;
    
    ASSERT_EQ(metadata.name, "test_model");
    ASSERT_EQ(metadata.architecture, "llama");
    ASSERT_EQ(metadata.version, "1.0");
    ASSERT_EQ(metadata.vocab_size, 32000);
    ASSERT_EQ(metadata.hidden_size, 4096);
    ASSERT_EQ(metadata.num_layers, 32);
    ASSERT_EQ(metadata.num_heads, 32);
    
    teardown_test();
}

void test_get_model_info_nonexistent() {
    std::cout << "\n--- Test: Get Model Info - Non-existent File ---" << std::endl;
    setup_test();
    
    // Test with non-existent file
    ASSERT_THROW(
        turboinfer::model::ModelLoader::get_model_info("non_existent_file.gguf"),
        std::runtime_error
    );
    
    teardown_test();
}

void test_format_detection() {
    std::cout << "\n--- Test: Model Format Detection ---" << std::endl;
    setup_test();
    
    // Create dummy files with different extensions
    auto gguf_file = test_dir / "test.gguf";
    auto safetensors_file = test_dir / "test.safetensors";
    auto pytorch_file = test_dir / "test.pth";
    auto onnx_file = test_dir / "test.onnx";
    auto unknown_file = test_dir / "test.unknown";

    // Create minimal files (they'll fail parsing but we can test format detection)
    std::ofstream(gguf_file) << "dummy";
    
    // For SafeTensors, create a file that's too small to have a valid header
    std::ofstream(safetensors_file) << "small"; // Only 5 bytes, but SafeTensors needs at least 8 for header size
    
    std::ofstream(pytorch_file) << "dummy";
    std::ofstream(onnx_file) << "dummy";
    std::ofstream(unknown_file) << "dummy";

    // Test format detection through error messages
    try {
        turboinfer::model::ModelLoader::get_model_info(gguf_file.string());
        ASSERT_TRUE(false); // Should have thrown
    } catch (const std::exception& e) {
        std::string error_msg = e.what();
        std::cout << "GGUF error: " << error_msg << std::endl;
        ASSERT_TRUE(error_msg.find("GGUF") != std::string::npos ||
                   error_msg.find("magic") != std::string::npos ||
                   error_msg.find("header") != std::string::npos ||
                   error_msg.find("Unable to read") != std::string::npos);
    }

    try {
        std::cout << "Testing SafeTensors file..." << std::endl;
        turboinfer::model::ModelLoader::get_model_info(safetensors_file.string());
        std::cout << "ERROR: SafeTensors should have thrown an exception!" << std::endl;
        ASSERT_TRUE(false); // Should have thrown
    } catch (const std::exception& e) {
        std::string error_msg = e.what();
        std::cout << "SafeTensors error: " << error_msg << std::endl;
        ASSERT_TRUE(error_msg.find("SafeTensors") != std::string::npos ||
                   error_msg.find("header") != std::string::npos ||
                   error_msg.find("Unable to read") != std::string::npos ||
                   error_msg.find("Failed to read") != std::string::npos ||
                   error_msg.find("Invalid") != std::string::npos);
    }

    // Test unknown format
    ASSERT_THROW(
        turboinfer::model::ModelLoader::get_model_info(unknown_file.string()),
        std::runtime_error
    );
    
    teardown_test();
}

void test_model_validation() {
    std::cout << "\n--- Test: Model Validation ---" << std::endl;
    setup_test();
    
    // Test with empty model data
    turboinfer::model::ModelData empty_model;
    turboinfer::model::ModelMetadata metadata;
    metadata.name = "test_model";
    metadata.architecture = "llama";
    metadata.version = "1.0";
    metadata.vocab_size = 32000;
    metadata.hidden_size = 4096;
    metadata.num_layers = 32;
    metadata.num_heads = 32;

    // This should return false for empty model
    bool result = turboinfer::model::ModelLoader::validate_model(empty_model, metadata);
    ASSERT_FALSE(result);

    // Test with basic model data containing essential tensors
    turboinfer::model::ModelData model_with_data;
    // Add a tensor with a name that matches the essential tensor patterns
    model_with_data.add_tensor("token_embeddings.weight", 
                              turboinfer::core::Tensor(turboinfer::core::TensorShape({32000, 4096})));
    model_with_data.add_tensor("norm.weight", 
                              turboinfer::core::Tensor(turboinfer::core::TensorShape({4096})));
    
    // This should return true for valid model with essential tensors
    bool valid_result = turboinfer::model::ModelLoader::validate_model(model_with_data, metadata);
    std::cout << "Validation result: " << (valid_result ? "true" : "false") << std::endl;
    
    // List tensor names for debugging
    auto tensor_names = model_with_data.tensor_names();
    std::cout << "Tensors in model: ";
    for (const auto& name : tensor_names) {
        std::cout << name << " ";
    }
    std::cout << std::endl;
    
    ASSERT_TRUE(valid_result);
    
    teardown_test();
}

void test_empty_file_handling() {
    std::cout << "\n--- Test: Empty File Handling ---" << std::endl;
    setup_test();
    
    auto empty_file_path = test_dir / "empty.gguf";
    std::ofstream empty_file_stream(empty_file_path); // Create empty file
    empty_file_stream.close();

    ASSERT_THROW(
        turboinfer::model::ModelLoader::get_model_info(empty_file_path.string()),
        std::runtime_error
    );
    
    teardown_test();
}

int main() {
    std::cout << "🚀 Starting test_model_loader Tests..." << std::endl;
    
    test_model_metadata();
    test_get_model_info_nonexistent();
    test_format_detection();
    test_model_validation();
    test_empty_file_handling();
    
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
