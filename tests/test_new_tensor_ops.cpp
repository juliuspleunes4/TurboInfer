/**
 * @file test_new_tensor_ops.cpp
 * @brief Tests for newly implemented tensor operations
 * @author J.J.G. Pleunes
 */

#include "turboinfer/core/tensor_engine.hpp"
#include "turboinfer/core/tensor.hpp"
#include <iostream>
#include <vector>
#include <cassert>

using namespace turboinfer::core;

#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        std::cerr << "ASSERTION FAILED: " << #condition << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } else { \
        std::cout << "✅ PASS: " << #condition << std::endl; \
    }

#define ASSERT_NEAR(a, b, tolerance) \
    if (std::abs((a) - (b)) > (tolerance)) { \
        std::cerr << "ASSERTION FAILED: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ") within " << (tolerance) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } else { \
        std::cout << "✅ PASS: " << #a << " ≈ " << #b << " within " << (tolerance) << std::endl; \
    }

void test_transpose() {
    std::cout << "\n--- Test: Transpose Operations ---" << std::endl;
    
    TensorEngine engine;
    
    // Test 2D transpose
    TensorShape shape_2d({2, 3});
    Tensor tensor_2d(shape_2d, DataType::kFloat32);
    
    // Fill with test data: [[1, 2, 3], [4, 5, 6]]
    float* data = tensor_2d.data_ptr<float>();
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    
    auto result = engine.transpose(tensor_2d);
    
    ASSERT_TRUE(result.shape().size(0) == 3);
    ASSERT_TRUE(result.shape().size(1) == 2);
    
    const float* result_data = result.data_ptr<float>();
    // Transposed should be: [[1, 4], [2, 5], [3, 6]]
    ASSERT_NEAR(result_data[0], 1.0f, 1e-6f);  // [0,0]
    ASSERT_NEAR(result_data[1], 4.0f, 1e-6f);  // [0,1]
    ASSERT_NEAR(result_data[2], 2.0f, 1e-6f);  // [1,0]
    ASSERT_NEAR(result_data[3], 5.0f, 1e-6f);  // [1,1]
    ASSERT_NEAR(result_data[4], 3.0f, 1e-6f);  // [2,0]
    ASSERT_NEAR(result_data[5], 6.0f, 1e-6f);  // [2,1]
    
    std::cout << "✅ 2D transpose test passed!" << std::endl;
}

void test_concatenate() {
    std::cout << "\n--- Test: Concatenate Operations ---" << std::endl;
    
    TensorEngine engine;
    
    // Test concatenation along dimension 0
    TensorShape shape1({2, 3});
    TensorShape shape2({1, 3});
    
    Tensor tensor1(shape1, DataType::kFloat32);
    Tensor tensor2(shape2, DataType::kFloat32);
    
    // Fill tensor1: [[1, 2, 3], [4, 5, 6]]
    float* data1 = tensor1.data_ptr<float>();
    for (int i = 0; i < 6; ++i) {
        data1[i] = static_cast<float>(i + 1);
    }
    
    // Fill tensor2: [[7, 8, 9]]
    float* data2 = tensor2.data_ptr<float>();
    for (int i = 0; i < 3; ++i) {
        data2[i] = static_cast<float>(i + 7);
    }
    
    std::vector<Tensor> tensors = {tensor1, tensor2};
    auto result = engine.concatenate(tensors, 0);
    
    ASSERT_TRUE(result.shape().size(0) == 3);
    ASSERT_TRUE(result.shape().size(1) == 3);
    
    const float* result_data = result.data_ptr<float>();
    // Should be: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for (int i = 0; i < 9; ++i) {
        ASSERT_NEAR(result_data[i], static_cast<float>(i + 1), 1e-6f);
    }
    
    std::cout << "✅ Concatenate test passed!" << std::endl;
}

void test_split() {
    std::cout << "\n--- Test: Split Operations ---" << std::endl;
    
    TensorEngine engine;
    
    // Create a tensor to split: shape (3, 4) = 12 elements
    TensorShape shape({3, 4});
    Tensor tensor(shape, DataType::kFloat32);
    
    float* data = tensor.data_ptr<float>();
    for (int i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    
    // Split along dimension 0 into sizes [1, 2]
    std::vector<size_t> split_sizes = {1, 2};
    auto results = engine.split(tensor, split_sizes, 0);
    
    ASSERT_TRUE(results.size() == 2);
    ASSERT_TRUE(results[0].shape().size(0) == 1);
    ASSERT_TRUE(results[0].shape().size(1) == 4);
    ASSERT_TRUE(results[1].shape().size(0) == 2);
    ASSERT_TRUE(results[1].shape().size(1) == 4);
    
    // Check first split: [[1, 2, 3, 4]]
    const float* data0 = results[0].data_ptr<float>();
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(data0[i], static_cast<float>(i + 1), 1e-6f);
    }
    
    // Check second split: [[5, 6, 7, 8], [9, 10, 11, 12]]
    const float* data1 = results[1].data_ptr<float>();
    for (int i = 0; i < 8; ++i) {
        ASSERT_NEAR(data1[i], static_cast<float>(i + 5), 1e-6f);
    }
    
    std::cout << "✅ Split test passed!" << std::endl;
}

void test_permute() {
    std::cout << "\n--- Test: Permute Operations ---" << std::endl;
    
    TensorEngine engine;
    
    // Create a 3D tensor: shape (2, 3, 4)
    TensorShape shape({2, 3, 4});
    Tensor tensor(shape, DataType::kFloat32);
    
    float* data = tensor.data_ptr<float>();
    for (int i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    
    // Permute dimensions: (2, 3, 4) -> (4, 2, 3) using permutation [2, 0, 1]
    std::vector<size_t> dims = {2, 0, 1};
    auto result = engine.permute(tensor, dims);
    
    ASSERT_TRUE(result.shape().size(0) == 4);
    ASSERT_TRUE(result.shape().size(1) == 2);
    ASSERT_TRUE(result.shape().size(2) == 3);
    
    // Verify the permutation worked (basic check)
    ASSERT_TRUE(result.shape().total_size() == 24);
    
    std::cout << "✅ Permute test passed!" << std::endl;
}

void test_multidimensional_slice() {
    std::cout << "\n--- Test: Multi-dimensional Slice Operations ---" << std::endl;
    
    // Create a 3D tensor: shape (2, 3, 4)
    TensorShape shape({2, 3, 4});
    Tensor tensor(shape, DataType::kFloat32);
    
    float* data = tensor.data_ptr<float>();
    for (int i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    
    // Slice: start=[0, 1, 1], end=[1, 3, 3] -> shape (1, 2, 2)
    std::vector<size_t> start = {0, 1, 1};
    std::vector<size_t> end = {1, 3, 3};
    
    auto result = tensor.slice(start, end);
    
    ASSERT_TRUE(result.shape().size(0) == 1);
    ASSERT_TRUE(result.shape().size(1) == 2);
    ASSERT_TRUE(result.shape().size(2) == 2);
    ASSERT_TRUE(result.shape().total_size() == 4);
    
    std::cout << "✅ Multi-dimensional slice test passed!" << std::endl;
}

int main() {
    std::cout << "🚀 Testing New Tensor Operations" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        test_transpose();
        test_concatenate();
        test_split();
        test_permute();
        test_multidimensional_slice();
        
        std::cout << "\n🎉 ALL NEW TENSOR OPERATIONS TESTS PASSED!" << std::endl;
        std::cout << "✅ transpose(), concatenate(), split(), permute(), and multi-dim slicing work correctly!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
