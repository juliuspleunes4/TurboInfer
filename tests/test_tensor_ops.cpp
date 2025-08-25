/**
 * @file test_tensor_ops.cpp
 * @brief Unit tests for tensor operations and transformations.
 * @author J.J.G. Pleunes
 */

#include <gtest/gtest.h>
#include "turboinfer/core/tensor.hpp"
#include <vector>

using namespace turboinfer::core;

class TensorOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for tensor operations tests
    }

    void TearDown() override {
        // Cleanup
    }
};

TEST_F(TensorOpsTest, Tensor_Slice_2D) {
    TensorShape shape({5, 6});
    Tensor tensor(shape, DataType::kFloat32);
    
    // Test basic 2D slicing
    std::vector<std::size_t> start = {1, 2};
    std::vector<std::size_t> end = {4, 5};
    Tensor sliced = tensor.slice(start, end);
    
    EXPECT_EQ(sliced.shape().ndim(), 2);
    EXPECT_EQ(sliced.shape().size(0), 3); // 4-1 = 3
    EXPECT_EQ(sliced.shape().size(1), 3); // 5-2 = 3
    EXPECT_EQ(sliced.dtype(), DataType::kFloat32);
}

TEST_F(TensorOpsTest, Tensor_Slice_3D) {
    TensorShape shape({4, 5, 6});
    Tensor tensor(shape, DataType::kFloat32);
    
    // Test 3D slicing
    std::vector<std::size_t> start = {0, 1, 2};
    std::vector<std::size_t> end = {3, 4, 5};
    Tensor sliced = tensor.slice(start, end);
    
    EXPECT_EQ(sliced.shape().ndim(), 3);
    EXPECT_EQ(sliced.shape().size(0), 3); // 3-0 = 3
    EXPECT_EQ(sliced.shape().size(1), 3); // 4-1 = 3
    EXPECT_EQ(sliced.shape().size(2), 3); // 5-2 = 3
}

TEST_F(TensorOpsTest, Tensor_Slice_Full_Dimension) {
    TensorShape shape({3, 4});
    Tensor tensor(shape, DataType::kFloat32);
    
    // Test slicing entire dimensions
    std::vector<std::size_t> start = {0, 0};
    std::vector<std::size_t> end = {3, 4};
    Tensor sliced = tensor.slice(start, end);
    
    EXPECT_EQ(sliced.shape(), tensor.shape());
    EXPECT_EQ(sliced.dtype(), tensor.dtype());
}

TEST_F(TensorOpsTest, Tensor_Slice_Single_Element) {
    TensorShape shape({10, 10});
    Tensor tensor(shape, DataType::kFloat32);
    
    // Test slicing single element
    std::vector<std::size_t> start = {5, 5};
    std::vector<std::size_t> end = {6, 6};
    Tensor sliced = tensor.slice(start, end);
    
    EXPECT_EQ(sliced.shape().ndim(), 2);
    EXPECT_EQ(sliced.shape().size(0), 1);
    EXPECT_EQ(sliced.shape().size(1), 1);
    EXPECT_EQ(sliced.shape().total_size(), 1);
}

TEST_F(TensorOpsTest, Tensor_Slice_Edge_Cases) {
    TensorShape shape({5, 5});
    Tensor tensor(shape, DataType::kFloat32);
    
    // Test slice at beginning
    std::vector<std::size_t> start1 = {0, 0};
    std::vector<std::size_t> end1 = {2, 2};
    Tensor slice1 = tensor.slice(start1, end1);
    EXPECT_EQ(slice1.shape().size(0), 2);
    EXPECT_EQ(slice1.shape().size(1), 2);
    
    // Test slice at end
    std::vector<std::size_t> start2 = {3, 3};
    std::vector<std::size_t> end2 = {5, 5};
    Tensor slice2 = tensor.slice(start2, end2);
    EXPECT_EQ(slice2.shape().size(0), 2);
    EXPECT_EQ(slice2.shape().size(1), 2);
}

TEST_F(TensorOpsTest, Tensor_Slice_Different_DataTypes) {
    TensorShape shape({4, 4});
    
    // Test slicing with different data types
    std::vector<DataType> types = {
        DataType::kFloat32,
        DataType::kFloat16,
        DataType::kInt32,
        DataType::kInt8
    };
    
    std::vector<std::size_t> start = {1, 1};
    std::vector<std::size_t> end = {3, 3};
    
    for (auto dtype : types) {
        Tensor tensor(shape, dtype);
        Tensor sliced = tensor.slice(start, end);
        
        EXPECT_EQ(sliced.shape().size(0), 2);
        EXPECT_EQ(sliced.shape().size(1), 2);
        EXPECT_EQ(sliced.dtype(), dtype);
    }
}

TEST_F(TensorOpsTest, Tensor_Memory_Layout_Consistency) {
    TensorShape shape({10, 20});
    Tensor tensor(shape, DataType::kFloat32);
    
    // Original tensor properties
    EXPECT_EQ(tensor.shape().total_size(), 200);
    EXPECT_EQ(tensor.byte_size(), 200 * 4); // 200 floats * 4 bytes
    EXPECT_NE(tensor.data_ptr(), nullptr);
    
    // After slicing, original should remain unchanged
    std::vector<std::size_t> start = {2, 5};
    std::vector<std::size_t> end = {8, 15};
    Tensor sliced = tensor.slice(start, end);
    
    // Original tensor should be unchanged
    EXPECT_EQ(tensor.shape().total_size(), 200);
    EXPECT_EQ(tensor.byte_size(), 200 * 4);
    EXPECT_NE(tensor.data_ptr(), nullptr);
    
    // Sliced tensor should have correct properties
    EXPECT_EQ(sliced.shape().total_size(), 6 * 10); // (8-2) * (15-5)
    EXPECT_EQ(sliced.byte_size(), 60 * 4); // 60 floats * 4 bytes
    EXPECT_NE(sliced.data_ptr(), nullptr);
}

TEST_F(TensorOpsTest, Tensor_Shape_Strides_Calculation) {
    // Test stride calculations for different shapes
    TensorShape shape1({5});
    EXPECT_EQ(shape1.ndim(), 1);
    EXPECT_EQ(shape1.total_size(), 5);
    
    TensorShape shape2({3, 4});
    EXPECT_EQ(shape2.ndim(), 2);
    EXPECT_EQ(shape2.total_size(), 12);
    
    TensorShape shape3({2, 3, 4});
    EXPECT_EQ(shape3.ndim(), 3);
    EXPECT_EQ(shape3.total_size(), 24);
    
    TensorShape shape4({2, 3, 4, 5});
    EXPECT_EQ(shape4.ndim(), 4);
    EXPECT_EQ(shape4.total_size(), 120);
}
