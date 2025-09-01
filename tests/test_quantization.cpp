/**
 * @file test_quantization.cpp
 * @brief Manual unit tests converted from GoogleTest.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <stdexcept>

// Test result tracking
int tests_run = 0;
int tests_passed = 0;

#define ASSERT_TRUE(condition) \
    do { \
        tests_run++; \
        if (condition) { \
            tests_passed++; \
            std::cout << "âœ… PASS: " << #condition << std::endl; \
        } else { \
            std::cout << "âŒ FAIL: " << #condition << std::endl; \
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
            std::cout << "âœ… PASS: " << #statement << " (no exception)" << std::endl; \
        } catch (...) { \
            std::cout << "âŒ FAIL: " << #statement << " (unexpected exception)" << std::endl; \
        } \
    } while(0)

#define ASSERT_THROW(statement, exception_type) \
    do { \
        tests_run++; \
        try { \
            statement; \
            std::cout << "âŒ FAIL: " << #statement << " (expected exception)" << std::endl; \
        } catch (const exception_type&) { \
            tests_passed++; \
            std::cout << "âœ… PASS: " << #statement << " (expected exception caught)" << std::endl; \
        } catch (...) { \
            std::cout << "âŒ FAIL: " << #statement << " (wrong exception type)" << std::endl; \
        } \
    } while(0)

void setup_test() {
    // Setup code - override in specific tests if needed
}

void teardown_test() {
    // Cleanup code - override in specific tests if needed
}

void test_quantization_operations() {
    std::cout << "\n--- Test: Quantization Operations ---" << std::endl;
    setup_test();
    
    try {
        turboinfer::optimize::Quantizer quantizer;
        
        // Create test tensor
        turboinfer::core::TensorShape shape({100, 200});
        turboinfer::core::Tensor test_tensor(shape, turboinfer::core::DataType::kFloat32);
        float* data = test_tensor.data_ptr<float>();
        
        // Fill with test data
        for (size_t i = 0; i < shape.total_size(); ++i) {
            data[i] = static_cast<float>(i % 256) - 128.0f; // Range: -128 to 127
        }
        
        // Test INT8 quantization
        auto int8_info = quantizer.calculate_quantization_info(test_tensor);
        
        ASSERT_TRUE(!int8_info.scales.empty());
        ASSERT_TRUE(int8_info.scales[0] > 0.0f);
        
        // Test compression ratio calculation for model data
        turboinfer::model::ModelData test_model;
        test_model.add_tensor("test_weights", test_tensor);
        float compression_ratio = quantizer.estimate_compression_ratio(test_model);
        ASSERT_TRUE(compression_ratio >= 3.8f && compression_ratio <= 4.2f); // Should be ~4x for FP32->INT8
        
        // Test quantization operation
        auto quantized_tensor = quantizer.quantize_tensor(test_tensor);
        ASSERT_EQ(quantized_tensor.dtype(), turboinfer::core::DataType::kInt8);
        
        std::cout << "✅ INT8 quantization: " << compression_ratio << "x compression" << std::endl;
        
        // Test INT4 quantization  
        turboinfer::optimize::QuantizationConfig int4_config;
        int4_config.type = turboinfer::optimize::QuantizationType::kInt4;
        turboinfer::optimize::Quantizer int4_quantizer(int4_config);
        
        auto int4_info = int4_quantizer.calculate_quantization_info(test_tensor);
        ASSERT_TRUE(!int4_info.scales.empty());
        ASSERT_TRUE(int4_info.scales[0] > 0.0f);
        
        turboinfer::model::ModelData int4_test_model;
        int4_test_model.add_tensor("test_weights", test_tensor);
        float int4_compression = int4_quantizer.estimate_compression_ratio(int4_test_model);
        ASSERT_TRUE(int4_compression >= 7.8f && int4_compression <= 8.2f); // Should be ~8x for FP32->INT4
        
        std::cout << "✅ INT4 quantization: " << int4_compression << "x compression" << std::endl;
        
        std::cout << "✅ Quantization operations test passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Quantization test failed: " << e.what() << std::endl;
        ASSERT_TRUE(false);
    }
    
    teardown_test();
}

void test_quantization_model_support() {
    std::cout << "\n--- Test: Quantization Model Support ---" << std::endl;
    setup_test();
    
    try {
        turboinfer::optimize::Quantizer quantizer;
        
        // Create mock model data
        turboinfer::model::ModelData model_data;
        turboinfer::model::ModelMetadata& metadata = model_data.metadata();
        metadata.vocab_size = 1000;
        metadata.hidden_size = 256;
        metadata.num_layers = 4;
        
        // Add some test tensors
        turboinfer::core::TensorShape weight_shape({256, 256});
        turboinfer::core::Tensor weight1(weight_shape, turboinfer::core::DataType::kFloat32);
        turboinfer::core::Tensor weight2(weight_shape, turboinfer::core::DataType::kFloat32);
        
        float* w1_data = weight1.data_ptr<float>();
        float* w2_data = weight2.data_ptr<float>();
        for (size_t i = 0; i < weight_shape.total_size(); ++i) {
            w1_data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
            w2_data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
        
        model_data.add_tensor("layers.0.attention.q_proj.weight", std::move(weight1));
        model_data.add_tensor("layers.0.attention.k_proj.weight", std::move(weight2));
        
        // Test model quantization
        auto quantized_model = quantizer.quantize_model(model_data);
        ASSERT_TRUE(quantized_model.tensor_names().size() >= 2);
        
        std::cout << "✅ Model quantization test passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Model quantization test failed: " << e.what() << std::endl;
        ASSERT_TRUE(false);
    }
    
    teardown_test();
}

int main() {
    std::cout << "ðŸš€ Starting test_quantization Tests..." << std::endl;
    
    test_quantization_operations();
    test_quantization_model_support();
    
    std::cout << "\nðŸ“Š Test Results:" << std::endl;
    std::cout << "Tests run: " << tests_run << std::endl;
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << (tests_run - tests_passed) << std::endl;
    
    if (tests_passed == tests_run) {
        std::cout << "ðŸŽ‰ ALL TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "âŒ SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}
