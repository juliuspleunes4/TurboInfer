/**
 * @file test_memory.cpp
 * @brief Unit tests for memory management and RAII patterns.
 * @author J.J.G. Pleunes
 */

#include <gtest/gtest.h>
#include "turboinfer/core/tensor.hpp"
#include <vector>
#include <memory>

using namespace turboinfer::core;

class MemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for memory tests
    }

    void TearDown() override {
        // Cleanup
    }
};

TEST_F(MemoryTest, Tensor_RAII_Basic) {
    // Test basic RAII - tensor should clean up automatically
    {
        TensorShape shape({1000, 1000}); // Large tensor
        Tensor tensor(shape, DataType::kFloat32);
        EXPECT_NE(tensor.data_ptr(), nullptr);
        EXPECT_EQ(tensor.byte_size(), 1000 * 1000 * 4);
    }
    // Tensor should be destroyed here - no memory leaks
    SUCCEED();
}

TEST_F(MemoryTest, Tensor_Copy_Constructor) {
    TensorShape shape({10, 10});
    Tensor original(shape, DataType::kFloat32);
    
    // Test copy constructor
    Tensor copy(original);
    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.dtype(), original.dtype());
    EXPECT_EQ(copy.byte_size(), original.byte_size());
    
    // Should have different memory addresses
    EXPECT_NE(copy.data_ptr(), original.data_ptr());
}

TEST_F(MemoryTest, Tensor_Assignment_Operator) {
    TensorShape shape1({5, 5});
    TensorShape shape2({3, 3});
    
    Tensor tensor1(shape1, DataType::kFloat32);
    Tensor tensor2(shape2, DataType::kFloat32);
    
    // Test assignment
    tensor2 = tensor1;
    EXPECT_EQ(tensor2.shape(), shape1);
    EXPECT_EQ(tensor2.dtype(), DataType::kFloat32);
    EXPECT_EQ(tensor2.byte_size(), tensor1.byte_size());
    
    // Should have different memory addresses
    EXPECT_NE(tensor2.data_ptr(), tensor1.data_ptr());
}

TEST_F(MemoryTest, Tensor_Move_Constructor) {
    TensorShape shape({100, 100});
    Tensor original(shape, DataType::kFloat32);
    void* original_ptr = original.data_ptr();
    
    // Test move constructor
    Tensor moved(std::move(original));
    EXPECT_EQ(moved.shape(), shape);
    EXPECT_EQ(moved.dtype(), DataType::kFloat32);
    EXPECT_EQ(moved.data_ptr(), original_ptr); // Should be same pointer
    
    // Original should be in valid but unspecified state
    EXPECT_EQ(original.data_ptr(), nullptr);
}

TEST_F(MemoryTest, Tensor_Move_Assignment) {
    TensorShape shape1({50, 50});
    TensorShape shape2({30, 30});
    
    Tensor tensor1(shape1, DataType::kFloat32);
    Tensor tensor2(shape2, DataType::kFloat32);
    void* tensor1_ptr = tensor1.data_ptr();
    
    // Test move assignment
    tensor2 = std::move(tensor1);
    EXPECT_EQ(tensor2.shape(), shape1);
    EXPECT_EQ(tensor2.data_ptr(), tensor1_ptr); // Should be same pointer
    
    // Original should be in valid but unspecified state
    EXPECT_EQ(tensor1.data_ptr(), nullptr);
}

TEST_F(MemoryTest, Multiple_Large_Tensors) {
    // Test creating multiple large tensors to check memory management
    std::vector<std::unique_ptr<Tensor>> tensors;
    
    for (int i = 0; i < 10; ++i) {
        TensorShape shape({100, 100});
        auto tensor = std::make_unique<Tensor>(shape, DataType::kFloat32);
        EXPECT_NE(tensor->data_ptr(), nullptr);
        EXPECT_EQ(tensor->byte_size(), 100 * 100 * 4);
        tensors.push_back(std::move(tensor));
    }
    
    // All tensors should be valid
    for (const auto& tensor : tensors) {
        EXPECT_NE(tensor->data_ptr(), nullptr);
        EXPECT_EQ(tensor->byte_size(), 100 * 100 * 4);
    }
    
    // Clear should clean up all memory
    tensors.clear();
    SUCCEED();
}

TEST_F(MemoryTest, Tensor_Slice_Memory_Management) {
    TensorShape shape({10, 10});
    Tensor original(shape, DataType::kFloat32);
    
    // Create slice
    std::vector<std::size_t> start = {0, 0};
    std::vector<std::size_t> end = {5, 5};
    Tensor sliced = original.slice(start, end);
    
    // Both tensors should be valid
    EXPECT_NE(original.data_ptr(), nullptr);
    EXPECT_NE(sliced.data_ptr(), nullptr);
    
    // Slice should have different shape but valid memory
    EXPECT_EQ(sliced.shape().size(0), 5);
    EXPECT_EQ(sliced.shape().size(1), 5);
}

TEST_F(MemoryTest, Tensor_Self_Assignment) {
    TensorShape shape({20, 20});
    Tensor tensor(shape, DataType::kFloat32);
    void* original_ptr = tensor.data_ptr();
    
    // Test self-assignment
    tensor = tensor;
    EXPECT_EQ(tensor.data_ptr(), original_ptr);
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.dtype(), DataType::kFloat32);
}
