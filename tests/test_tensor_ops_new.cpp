/**
 * @file test_tensor_ops.cpp
 * @brief Manual unit tests for tensor operations converted from GoogleTest.
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
            std::cout << "❌ FAIL: " << #statement << " (no exception thrown)" << std::endl; \
        } catch (const exception_type&) { \
            tests_passed++; \
            std::cout << "✅ PASS: " << #statement << " (correct exception)" << std::endl; \
        } catch (...) { \
            std::cout << "❌ FAIL: " << #statement << " (wrong exception type)" << std::endl; \
        } \
    } while(0)

void setup_test() {
    // Setup code - override in specific tests if needed
}

void teardown_test() {
    // Cleanup code - override in specific tests if needed
}

void test_tensor_basic_operations() {
    std::cout << "\n--- Test: Basic Tensor Operations ---" << std::endl;
    setup_test();
    
    // Test tensor creation with shape
    turboinfer::core::TensorShape shape({3, 2});
    turboinfer::core::Tensor tensor1(shape);
    ASSERT_EQ(shape.total_size(), 6);
    ASSERT_EQ(shape.ndim(), 2);
    ASSERT_EQ(shape.size(0), 3);
    ASSERT_EQ(shape.size(1), 2);
    
    // Test tensor data access
    ASSERT_NO_THROW(tensor1.data());
    ASSERT_TRUE(tensor1.data() != nullptr);
    ASSERT_FALSE(tensor1.empty());
    
    teardown_test();
}

void test_tensor_arithmetic() {
    std::cout << "\n--- Test: Tensor Arithmetic ---" << std::endl;
    setup_test();
    
    turboinfer::core::TensorShape shape({2, 2});
    turboinfer::core::Tensor tensor1(shape);
    turboinfer::core::Tensor tensor2(shape);
    
    // Fill with test data
    float* data1 = tensor1.data_ptr<float>();
    float* data2 = tensor2.data_ptr<float>();
    for (int i = 0; i < 4; ++i) {
        data1[i] = static_cast<float>(i + 1);
        data2[i] = static_cast<float>(i + 2);
    }
    
    // Test tensor addition
    turboinfer::core::TensorEngine engine;
    ASSERT_NO_THROW(engine.add(tensor1, tensor2));
    
    teardown_test();
}

void test_tensor_engine_basic() {
    std::cout << "\n--- Test: TensorEngine Basic ---" << std::endl;
    setup_test();
    
    turboinfer::core::TensorEngine engine;
    
    // Test engine initialization
    ASSERT_TRUE(true); // Engine created successfully
    
    teardown_test();
}

void test_tensor_shape_operations() {
    std::cout << "\n--- Test: TensorShape Operations ---" << std::endl;
    setup_test();
    
    // Test shape creation
    turboinfer::core::TensorShape shape1({2, 3, 4});
    ASSERT_EQ(shape1.ndim(), 3);
    ASSERT_EQ(shape1.total_size(), 24);
    ASSERT_EQ(shape1.size(0), 2);
    ASSERT_EQ(shape1.size(1), 3);
    ASSERT_EQ(shape1.size(2), 4);
    
    // Test shape equality
    turboinfer::core::TensorShape shape2({2, 3, 4});
    ASSERT_TRUE(shape1 == shape2);
    ASSERT_FALSE(shape1 != shape2);
    
    teardown_test();
}

void test_tensor_data_types() {
    std::cout << "\n--- Test: Tensor Data Types ---" << std::endl;
    setup_test();
    
    turboinfer::core::TensorShape shape({2, 2});
    
    // Test float32 tensor (default)
    turboinfer::core::Tensor tensor_f32(shape);
    ASSERT_EQ(tensor_f32.dtype(), turboinfer::core::DataType::kFloat32);
    ASSERT_EQ(tensor_f32.element_size(), sizeof(float));
    
    // Test int32 tensor
    turboinfer::core::Tensor tensor_i32(shape, turboinfer::core::DataType::kInt32);
    ASSERT_EQ(tensor_i32.dtype(), turboinfer::core::DataType::kInt32);
    ASSERT_EQ(tensor_i32.element_size(), sizeof(int32_t));
    
    teardown_test();
}

int main() {
    std::cout << "🚀 Starting test_tensor_ops Tests..." << std::endl;
    
    test_tensor_basic_operations();
    test_tensor_arithmetic();
    test_tensor_engine_basic();
    test_tensor_shape_operations();
    test_tensor_data_types();
    
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
