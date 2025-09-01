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
        } else { \
            std::cerr << "ASSERTION FAILED: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
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
    turboinfer::core::TensorShape shape1d({10});
    turboinfer::core::Tensor tensor1d(shape1d);
    ASSERT_TRUE(tensor1d.shape().ndim() == 1);
    ASSERT_TRUE(tensor1d.shape().total_size() == 10);
    ASSERT_TRUE(tensor1d.shape().size(0) == 10);
    
    // Test 2D tensor creation
    turboinfer::core::TensorShape shape2d({3, 4});
    turboinfer::core::Tensor tensor2d(shape2d);
    ASSERT_TRUE(tensor2d.shape().ndim() == 2);
    ASSERT_TRUE(tensor2d.shape().total_size() == 12);
    ASSERT_TRUE(tensor2d.shape().size(0) == 3 && tensor2d.shape().size(1) == 4);
    
    // Test 3D tensor creation
    turboinfer::core::TensorShape shape3d({2, 3, 4});
    turboinfer::core::Tensor tensor3d(shape3d);
    ASSERT_TRUE(tensor3d.shape().ndim() == 3);
    ASSERT_TRUE(tensor3d.shape().total_size() == 24);
    ASSERT_TRUE(tensor3d.shape().size(0) == 2 && tensor3d.shape().size(1) == 3 && tensor3d.shape().size(2) == 4);
    
    std::cout << "✅ Tensor creation tests passed" << std::endl;
    teardown_test();
}

void test_tensor_data_access() {
    std::cout << "\n--- Test: Tensor Data Access ---" << std::endl;
    setup_test();
    
    // Create a 2x3 tensor
    turboinfer::core::TensorShape shape({2, 3});
    turboinfer::core::Tensor tensor(shape);
    
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
    turboinfer::core::TensorShape original_shape({2, 6});
    turboinfer::core::Tensor tensor(original_shape);
    
    // Fill with test data
    float* data = tensor.data_ptr<float>();
    for (size_t i = 0; i < tensor.shape().total_size(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Reshape to 3x4
    turboinfer::core::TensorShape new_shape({3, 4});
    turboinfer::core::Tensor reshaped = tensor.reshape(new_shape);
    
    ASSERT_TRUE(reshaped.shape().dimensions() == new_shape.dimensions());
    ASSERT_TRUE(reshaped.shape().ndim() == 2);
    ASSERT_TRUE(reshaped.shape().total_size() == 12);
    
    // Verify data is preserved
    float* reshaped_data = reshaped.data_ptr<float>();
    for (size_t i = 0; i < reshaped.shape().total_size(); ++i) {
        ASSERT_TRUE(reshaped_data[i] == static_cast<float>(i));
    }
    
    std::cout << "✅ Tensor reshape tests passed" << std::endl;
    teardown_test();
}

void test_tensor_slice() {
    std::cout << "\n--- Test: Tensor Slice ---" << std::endl;
    setup_test();
    
    // Create a 4x4 tensor
    turboinfer::core::TensorShape shape({4, 4});
    turboinfer::core::Tensor tensor(shape);
    
    // Fill with test data
    float* data = tensor.data_ptr<float>();
    for (size_t i = 0; i < tensor.shape().total_size(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Create a slice [1:3, 1:3] (2x2 submatrix)
    std::vector<size_t> start = {1, 1};
    std::vector<size_t> end = {3, 3};
    turboinfer::core::Tensor sliced = tensor.slice(start, end);
    
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
    turboinfer::core::TensorShape shape({2, 3});
    turboinfer::core::Tensor original(shape);
    
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

void test_tensor_memory_management() {
    std::cout << "\n--- Test: Tensor Memory Management ---" << std::endl;
    setup_test();
    
    // Test multiple tensor creation and destruction
    std::vector<std::unique_ptr<turboinfer::core::Tensor>> tensors;
    
    for (int i = 1; i <= 10; ++i) {
        turboinfer::core::TensorShape shape({static_cast<size_t>(i), static_cast<size_t>(i)});
        auto tensor = std::make_unique<turboinfer::core::Tensor>(shape);
        ASSERT_TRUE(tensor->shape().total_size() == i * i);
        ASSERT_TRUE(tensor->data() != nullptr);
        tensors.push_back(std::move(tensor));
    }
    
    // Verify all tensors are valid
    ASSERT_TRUE(tensors.size() == 10);
    
    // Test that each tensor has different memory addresses
    for (size_t i = 1; i < tensors.size(); ++i) {
        ASSERT_TRUE(tensors[i]->data() != tensors[i-1]->data());
    }
    
    std::cout << "✅ Tensor memory management test passed" << std::endl;
    teardown_test();
}

int main() {
    std::cout << "🚀 Starting test_tensor Tests..." << std::endl;
    
    test_tensor_creation();
    test_tensor_data_access();
    test_tensor_reshape();
    test_tensor_slice();
    test_tensor_clone();
    test_tensor_memory_management();
    
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
