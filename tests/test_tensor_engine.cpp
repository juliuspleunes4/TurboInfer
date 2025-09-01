/**
 * @file test_tensor_engine.cpp
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

void test_tensor_engine_operations() {
    std::cout << "\n--- Test: TensorEngine Basic Operations ---" << std::endl;
    setup_test();
    
    try {
        turboinfer::core::TensorEngine engine(turboinfer::core::ComputeDevice::kCPU);
        
        // Test matrix multiplication
        turboinfer::core::TensorShape shape_a({2, 3});
        turboinfer::core::TensorShape shape_b({3, 4});
        turboinfer::core::Tensor a(shape_a, turboinfer::core::DataType::kFloat32);
        turboinfer::core::Tensor b(shape_b, turboinfer::core::DataType::kFloat32);
        
        // Initialize test data
        float* a_data = a.data_ptr<float>();
        float* b_data = b.data_ptr<float>();
        
        for (size_t i = 0; i < 6; ++i) a_data[i] = static_cast<float>(i + 1);
        for (size_t i = 0; i < 12; ++i) b_data[i] = static_cast<float>(i + 1);
        
        // Test GEMM operation
        auto result = engine.matmul(a, b);
        ASSERT_TRUE(result.shape().size(0) == 2);
        ASSERT_TRUE(result.shape().size(1) == 4);
        
        // Test addition
        turboinfer::core::TensorShape add_shape({2, 2});
        turboinfer::core::Tensor x(add_shape, turboinfer::core::DataType::kFloat32);
        turboinfer::core::Tensor y(add_shape, turboinfer::core::DataType::kFloat32);
        
        float* x_data = x.data_ptr<float>();
        float* y_data = y.data_ptr<float>();
        for (size_t i = 0; i < 4; ++i) {
            x_data[i] = static_cast<float>(i);
            y_data[i] = static_cast<float>(i + 1);
        }
        
        auto sum = engine.add(x, y);
        const float* sum_data = sum.data_ptr<float>();
        for (size_t i = 0; i < 4; ++i) {
            ASSERT_TRUE(std::abs(sum_data[i] - (2 * i + 1)) < 1e-6f);
        }
        
        // Test softmax
        turboinfer::core::TensorShape softmax_shape({1, 4});
        turboinfer::core::Tensor logits(softmax_shape, turboinfer::core::DataType::kFloat32);
        float* logits_data = logits.data_ptr<float>();
        logits_data[0] = 1.0f; logits_data[1] = 2.0f; 
        logits_data[2] = 3.0f; logits_data[3] = 4.0f;
        
        auto probs = engine.softmax(logits);
        const float* probs_data = probs.data_ptr<float>();
        
        // Check probabilities sum to 1
        float prob_sum = 0.0f;
        for (size_t i = 0; i < 4; ++i) {
            prob_sum += probs_data[i];
            ASSERT_TRUE(probs_data[i] > 0.0f); // All positive
        }
        ASSERT_TRUE(std::abs(prob_sum - 1.0f) < 1e-6f);
        
        std::cout << "✅ TensorEngine operations test passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ TensorEngine test failed: " << e.what() << std::endl;
        ASSERT_TRUE(false);
    }
    
    teardown_test();
}

void test_tensor_engine_device_info() {
    std::cout << "\n--- Test: TensorEngine Device Information ---" << std::endl;
    setup_test();
    
    try {
        turboinfer::core::TensorEngine engine(turboinfer::core::ComputeDevice::kAuto);
        
        // Test device info
        std::string device_info = engine.device_info();
        ASSERT_TRUE(!device_info.empty());
        std::cout << "Device info: " << device_info << std::endl;
        
        // Test GPU availability check
        bool gpu_available = engine.gpu_available();
        std::cout << "GPU available: " << (gpu_available ? "Yes" : "No") << std::endl;
        
        std::cout << "✅ TensorEngine device info test passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ TensorEngine device info test failed: " << e.what() << std::endl;
        ASSERT_TRUE(false);
    }
    
    teardown_test();
}

int main() {
    std::cout << "ðŸš€ Starting test_tensor_engine Tests..." << std::endl;
    
    test_tensor_engine_operations();
    test_tensor_engine_device_info();
    
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
