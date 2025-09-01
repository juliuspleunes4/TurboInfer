/**
 * @file test_tensor.cpp
 * @brief Manual unit tests converted from GoogleTest.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

// Global test counters
int tests_run = 0;
int tests_passed = 0;

// Simple assertion macro
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
    turboinfer::core::TensorShape tensor_shape1d(shape1d);
    turboinfer::core::Tensor tensor1d(tensor_shape1d);
    ASSERT_TRUE(tensor1d.shape().ndim() == 1);
    ASSERT_TRUE(tensor1d.shape().total_size() == 10);
    ASSERT_TRUE(tensor1d.shape().dimensions() == shape1d);
    
    // Test 2D tensor creation
    std::vector<size_t> shape2d = {3, 4};
    turboinfer::core::TensorShape tensor_shape2d(shape2d);
    turboinfer::core::Tensor tensor2d(tensor_shape2d);
    ASSERT_TRUE(tensor2d.shape().ndim() == 2);
    ASSERT_TRUE(tensor2d.shape().total_size() == 12);
    ASSERT_TRUE(tensor2d.shape().dimensions() == shape2d);
    
    // Test 3D tensor creation
    std::vector<size_t> shape3d = {2, 3, 4};
    turboinfer::core::TensorShape tensor_shape3d(shape3d);
    turboinfer::core::Tensor tensor3d(tensor_shape3d);
    ASSERT_TRUE(tensor3d.shape().ndim() == 3);
    ASSERT_TRUE(tensor3d.shape().total_size() == 24);
    ASSERT_TRUE(tensor3d.shape().dimensions() == shape3d);
    
    std::cout << "✅ Tensor creation tests passed" << std::endl;
    teardown_test();
}

void test_tensor_data_access() {
    std::cout << "\n--- Test: Tensor Data Access ---" << std::endl;
    setup_test();
    
    // Create a 2x3 tensor
    std::vector<size_t> shape = {2, 3};
    turboinfer::core::TensorShape tensor_shape(shape);
    turboinfer::core::Tensor tensor(tensor_shape);
    
    // Test data access and modification
    float* data = tensor.data_ptr<float>();
    ASSERT_TRUE(data != nullptr);
    
    // Set some values
    for (size_t i = 0; i < tensor.shape().total_size(); ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    
    // Verify values
    for (size_t i = 0; i < tensor.shape().total_size(); ++i) {
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
    turboinfer::core::TensorShape tensor_shape(original_shape);
    turboinfer::core::Tensor tensor(tensor_shape);
    
    // Fill with test data
    float* data = tensor.data_ptr<float>();
    for (size_t i = 0; i < tensor.shape().total_size(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Reshape to 3x4
    std::vector<size_t> new_shape = {3, 4};
    turboinfer::core::TensorShape new_tensor_shape(new_shape);
    tensor = tensor.reshape(new_tensor_shape);
    
    ASSERT_TRUE(tensor.shape().dimensions() == new_shape);
    ASSERT_TRUE(tensor.shape().ndim() == 2);
    ASSERT_TRUE(tensor.shape().total_size() == 12);
    
    // Verify data is preserved
    data = tensor.data_ptr<float>();
    for (size_t i = 0; i < tensor.shape().total_size(); ++i) {
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
    turboinfer::core::TensorShape tensor_shape(shape);
    turboinfer::core::Tensor tensor(tensor_shape);
    
    // Fill with test data
    float* data = tensor.data_ptr<float>();
    for (size_t i = 0; i < tensor.shape().total_size(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Create a slice [1:3, 1:3] (2x2 submatrix)
    std::vector<size_t> start_indices = {1, 1};
    std::vector<size_t> end_indices = {3, 3};
    turboinfer::core::Tensor sliced = tensor.slice(start_indices, end_indices);
    
    ASSERT_TRUE(sliced.shape().ndim() == 2);
    ASSERT_TRUE(sliced.shape().size(0) == 2);
    ASSERT_TRUE(sliced.shape().size(1) == 2);
    ASSERT_TRUE(sliced.shape().total_size() == 4);
    
    std::cout << "✅ Tensor slice tests passed" << std::endl;
    teardown_test();
}

void test_tensor_clone() {
    std::cout << "\n--- Test: Tensor Clone ---" << std::endl;
    setup_test();
    
    // Create original tensor
    std::vector<size_t> shape = {2, 3};
    turboinfer::core::TensorShape tensor_shape(shape);
    turboinfer::core::Tensor original(tensor_shape);
    
    // Fill with test data
    float* orig_data = original.data_ptr<float>();
    for (size_t i = 0; i < original.shape().total_size(); ++i) {
        orig_data[i] = static_cast<float>(i * 2);
    }
    
    // Clone the tensor
    turboinfer::core::Tensor clone = original.clone();
    
    // Verify clone has same properties
    ASSERT_TRUE(clone.shape().dimensions() == original.shape().dimensions());
    ASSERT_TRUE(clone.shape().ndim() == original.shape().ndim());
    ASSERT_TRUE(clone.shape().total_size() == original.shape().total_size());
    
    // Verify data is copied
    float* clone_data = clone.data_ptr<float>();
    for (size_t i = 0; i < clone.shape().total_size(); ++i) {
        ASSERT_TRUE(clone_data[i] == orig_data[i]);
    }
    
    // Verify they are independent (modify original)
    orig_data[0] = 999.0f;
    ASSERT_TRUE(clone_data[0] != orig_data[0]);
    
    std::cout << "✅ Tensor clone tests passed" << std::endl;
    teardown_test();
}

void test_tensor_advanced_operations() {
    std::cout << "\n--- Test: Advanced Tensor Operations ---" << std::endl;
    setup_test();
    
    // Test tensor broadcasting concepts
    turboinfer::core::TensorShape shape1({2, 3});
    turboinfer::core::TensorShape shape2({1, 3});
    turboinfer::core::Tensor tensor1(shape1);
    turboinfer::core::Tensor tensor2(shape2);
    
    ASSERT_EQ(shape1.total_size(), 6);
    ASSERT_EQ(shape2.total_size(), 3);
    
    // Test tensor memory layout
    ASSERT_TRUE(tensor1.data() != nullptr);
    ASSERT_TRUE(tensor2.data() != nullptr);
    ASSERT_TRUE(tensor1.data() != tensor2.data()); // Different memory locations
    
    // Test tensor properties
    ASSERT_FALSE(tensor1.empty());
    ASSERT_FALSE(tensor2.empty());
    ASSERT_EQ(tensor1.shape().ndim(), 2);
    ASSERT_EQ(tensor2.shape().ndim(), 2);
    
    std::cout << "✅ Advanced tensor operations test passed" << std::endl;
    teardown_test();
}

void test_tensor_edge_cases() {
    std::cout << "\n--- Test: Tensor Edge Cases ---" << std::endl;
    setup_test();
    
    // Test single element tensor
    turboinfer::core::TensorShape single_shape({1});
    turboinfer::core::Tensor single_tensor(single_shape);
    ASSERT_EQ(single_tensor.shape().total_size(), 1);
    ASSERT_EQ(single_tensor.shape().ndim(), 1);
    
    // Test large dimension tensor
    turboinfer::core::TensorShape large_shape({10, 20, 5});
    turboinfer::core::Tensor large_tensor(large_shape);
    ASSERT_EQ(large_tensor.shape().total_size(), 1000);
    ASSERT_EQ(large_tensor.shape().ndim(), 3);
    
    // Test tensor with same shape comparison
    turboinfer::core::TensorShape shape_a({3, 4});
    turboinfer::core::TensorShape shape_b({3, 4});
    ASSERT_TRUE(shape_a == shape_b);
    ASSERT_FALSE(shape_a != shape_b);
    
    std::cout << "✅ Tensor edge cases test passed" << std::endl;
    teardown_test();
}

int main() {
    std::cout << "🚀 Starting test_tensor Tests..." << std::endl;
    
    test_tensor_creation();
    test_tensor_data_access();
    test_tensor_reshape();
    test_tensor_slice();
    test_tensor_clone();
    test_tensor_advanced_operations();
    test_tensor_edge_cases();
    
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
