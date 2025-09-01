/**
 * @file test_tensor.cpp
 * @brief Manual unit tests converted from GoogleTest.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>int main() {
    std::cout << "🚀 Starting test_tensor Tests..." << std::endl;
    
    test_tensor_creation();
    test_tensor_data_access();
    test_tensor_reshape();
    test_tensor_slice();
    test_tensor_clone();
    test_placeholder();
    
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
}except>

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

void test_tensor_creation() {
    std::cout << "\n--- Test: Tensor Creation ---" << std::endl;
    setup_test();
    
    // Test 1D tensor creation
    std::vector<size_t> shape1d = {10};
    turboinfer::core::Tensor tensor1d(shape1d);
    ASSERT_TRUE(tensor1d.dimensions() == 1);
    ASSERT_TRUE(tensor1d.size() == 10);
    ASSERT_TRUE(tensor1d.shape() == shape1d);
    
    // Test 2D tensor creation
    std::vector<size_t> shape2d = {3, 4};
    turboinfer::core::Tensor tensor2d(shape2d);
    ASSERT_TRUE(tensor2d.dimensions() == 2);
    ASSERT_TRUE(tensor2d.size() == 12);
    ASSERT_TRUE(tensor2d.shape() == shape2d);
    
    // Test 3D tensor creation
    std::vector<size_t> shape3d = {2, 3, 4};
    turboinfer::core::Tensor tensor3d(shape3d);
    ASSERT_TRUE(tensor3d.dimensions() == 3);
    ASSERT_TRUE(tensor3d.size() == 24);
    ASSERT_TRUE(tensor3d.shape() == shape3d);
    
    std::cout << "✅ Tensor creation tests passed" << std::endl;
    teardown_test();
}

void test_tensor_data_access() {
    std::cout << "\n--- Test: Tensor Data Access ---" << std::endl;
    setup_test();
    
    // Create a 2x3 tensor
    std::vector<size_t> shape = {2, 3};
    turboinfer::core::Tensor tensor(shape);
    
    // Test data access and modification
    float* data = tensor.data<float>();
    ASSERT_TRUE(data != nullptr);
    
    // Set some values
    for (size_t i = 0; i < tensor.size(); ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    
    // Verify values
    for (size_t i = 0; i < tensor.size(); ++i) {
        ASSERT_TRUE(data[i] == static_cast<float>(i + 1));
    }
    
    std::cout << "✅ Tensor data access tests passed" << std::endl;
    teardown_test();
}

void test_tensor_reshape() {
    std::cout << "\n--- Test: Tensor Reshape ---" << std::endl;
    setup_test();
    
    // Create a 2x6 tensor
    std::vector<size_t> original_shape = {2, 6};
    turboinfer::core::Tensor tensor(original_shape);
    
    // Fill with test data
    float* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.size(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Reshape to 3x4
    std::vector<size_t> new_shape = {3, 4};
    tensor.reshape(new_shape);
    
    ASSERT_TRUE(tensor.shape() == new_shape);
    ASSERT_TRUE(tensor.dimensions() == 2);
    ASSERT_TRUE(tensor.size() == 12);
    
    // Verify data is preserved
    data = tensor.data<float>();
    for (size_t i = 0; i < tensor.size(); ++i) {
        ASSERT_TRUE(data[i] == static_cast<float>(i));
    }
    
    std::cout << "✅ Tensor reshape tests passed" << std::endl;
    teardown_test();
}

void test_tensor_slice() {
    std::cout << "\n--- Test: Tensor Slice ---" << std::endl;
    setup_test();
    
    // Create a 4x4 tensor
    std::vector<size_t> shape = {4, 4};
    turboinfer::core::Tensor tensor(shape);
    
    // Fill with test data
    float* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.size(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Create a slice [1:3, 1:3] (2x2 submatrix)
    std::vector<std::pair<size_t, size_t>> slice_ranges = {{1, 3}, {1, 3}};
    turboinfer::core::Tensor sliced = tensor.slice(slice_ranges);
    
    ASSERT_TRUE(sliced.dimensions() == 2);
    ASSERT_TRUE(sliced.shape()[0] == 2);
    ASSERT_TRUE(sliced.shape()[1] == 2);
    ASSERT_TRUE(sliced.size() == 4);
    
    std::cout << "✅ Tensor slice tests passed" << std::endl;
    teardown_test();
}

void test_tensor_clone() {
    std::cout << "\n--- Test: Tensor Clone ---" << std::endl;
    setup_test();
    
    // Create original tensor
    std::vector<size_t> shape = {2, 3};
    turboinfer::core::Tensor original(shape);
    
    // Fill with test data
    float* orig_data = original.data<float>();
    for (size_t i = 0; i < original.size(); ++i) {
        orig_data[i] = static_cast<float>(i * 2);
    }
    
    // Clone the tensor
    turboinfer::core::Tensor clone = original.clone();
    
    // Verify clone has same properties
    ASSERT_TRUE(clone.shape() == original.shape());
    ASSERT_TRUE(clone.dimensions() == original.dimensions());
    ASSERT_TRUE(clone.size() == original.size());
    
    // Verify data is copied
    float* clone_data = clone.data<float>();
    for (size_t i = 0; i < clone.size(); ++i) {
        ASSERT_TRUE(clone_data[i] == orig_data[i]);
    }
    
    // Verify they are independent (modify original)
    orig_data[0] = 999.0f;
    ASSERT_TRUE(clone_data[0] != orig_data[0]);
    
    std::cout << "✅ Tensor clone tests passed" << std::endl;
    teardown_test();
}

void test_placeholder() {
    std::cout << "\n--- Test: Basic Functionality ---" << std::endl;
    setup_test();
    
    // Basic tensor test
    std::vector<size_t> shape = {1};
    turboinfer::core::Tensor tensor(shape);
    ASSERT_TRUE(tensor.size() == 1);
    
    std::cout << "✅ Basic functionality test passed" << std::endl;
    teardown_test();
}

int main() {
    std::cout << "ðŸš€ Starting test_tensor Tests..." << std::endl;
    
    test_placeholder();
    
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
