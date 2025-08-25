/**
 * @file test_error_handling.cpp
 * @brief Unit tests for error handling and exception safety.
 * @author J.J.G. Pleunes
 */

#include <gtest/gtest.h>
#include "turboinfer/core/tensor.hpp"
#include "turboinfer/core/tensor_engine.hpp"
#include "turboinfer/model/model_loader.hpp"
#include <stdexcept>

using namespace turboinfer::core;
using namespace turboinfer::model;

class ErrorHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for error handling tests
    }

    void TearDown() override {
        // Cleanup
    }
};

TEST_F(ErrorHandlingTest, TensorShape_Invalid_Dimensions) {
    // Test invalid shape constructions
    EXPECT_THROW(TensorShape({}), std::invalid_argument); // Empty dimensions
    
    std::vector<size_t> invalid_dims = {0, 5}; // Zero dimension
    EXPECT_THROW(TensorShape(invalid_dims), std::invalid_argument);
    
    std::vector<size_t> very_large = {SIZE_MAX, SIZE_MAX}; // Overflow risk
    EXPECT_THROW(TensorShape(very_large), std::invalid_argument);
}

TEST_F(ErrorHandlingTest, TensorShape_Out_Of_Bounds_Access) {
    TensorShape shape({3, 4, 5});
    
    // Valid access
    EXPECT_NO_THROW(shape.size(0));
    EXPECT_NO_THROW(shape.size(1));
    EXPECT_NO_THROW(shape.size(2));
    
    // Invalid access
    EXPECT_THROW(shape.size(3), std::out_of_range);
    EXPECT_THROW(shape.size(10), std::out_of_range);
}

TEST_F(ErrorHandlingTest, Tensor_Invalid_Slice_Parameters) {
    TensorShape shape({5, 6});
    Tensor tensor(shape, DataType::kFloat32);
    
    // Test invalid slice parameters
    std::vector<std::size_t> start_invalid = {6, 0}; // Start beyond bounds
    std::vector<std::size_t> end_valid = {5, 6};
    EXPECT_THROW(tensor.slice(start_invalid, end_valid), std::out_of_range);
    
    std::vector<std::size_t> start_valid = {0, 0};
    std::vector<std::size_t> end_invalid = {5, 7}; // End beyond bounds
    EXPECT_THROW(tensor.slice(start_valid, end_invalid), std::out_of_range);
    
    // Start >= end
    std::vector<std::size_t> start_large = {3, 4};
    std::vector<std::size_t> end_small = {2, 3};
    EXPECT_THROW(tensor.slice(start_large, end_small), std::invalid_argument);
}

TEST_F(ErrorHandlingTest, Tensor_Mismatched_Slice_Dimensions) {
    TensorShape shape({5, 6});
    Tensor tensor(shape, DataType::kFloat32);
    
    // Wrong number of dimensions in slice parameters
    std::vector<std::size_t> start_wrong_dim = {0}; // Only 1D for 2D tensor
    std::vector<std::size_t> end_wrong_dim = {5};
    EXPECT_THROW(tensor.slice(start_wrong_dim, end_wrong_dim), std::invalid_argument);
    
    std::vector<std::size_t> start_too_many = {0, 0, 0}; // 3D for 2D tensor
    std::vector<std::size_t> end_too_many = {5, 6, 1};
    EXPECT_THROW(tensor.slice(start_too_many, end_too_many), std::invalid_argument);
    
    // Different sizes for start and end
    std::vector<std::size_t> start_2d = {0, 0};
    std::vector<std::size_t> end_1d = {5};
    EXPECT_THROW(tensor.slice(start_2d, end_1d), std::invalid_argument);
}

TEST_F(ErrorHandlingTest, TensorEngine_Invalid_Operations) {
    TensorEngine engine;
    
    // Test operations with mismatched dimensions
    TensorShape shape1({3, 4});
    TensorShape shape2({4, 5});
    TensorShape shape3({2, 3}); // Wrong size for multiplication result
    
    Tensor a(shape1, DataType::kFloat32);
    Tensor b(shape2, DataType::kFloat32);
    Tensor c(shape3, DataType::kFloat32);
    
    // This should work (3x4) * (4x5) = (3x5)
    TensorShape correct_shape({3, 5});
    Tensor correct_result(correct_shape, DataType::kFloat32);
    EXPECT_NO_THROW(engine.matrix_multiply(a, b, correct_result));
    
    // This should fail - wrong output dimensions
    EXPECT_THROW(engine.matrix_multiply(a, b, c), std::invalid_argument);
}

TEST_F(ErrorHandlingTest, TensorEngine_DataType_Mismatch) {
    TensorEngine engine;
    
    TensorShape shape({3, 3});
    Tensor a(shape, DataType::kFloat32);
    Tensor b(shape, DataType::kInt32); // Different data type
    Tensor c(shape, DataType::kFloat32);
    
    // Should fail due to data type mismatch
    EXPECT_THROW(engine.matrix_multiply(a, b, c), std::invalid_argument);
}

TEST_F(ErrorHandlingTest, ModelLoader_Invalid_File_Path) {
    ModelLoader loader;
    
    // Test with non-existent file
    EXPECT_THROW(loader.load_gguf_model("non_existent_file.gguf"), std::runtime_error);
    
    // Test with empty path
    EXPECT_THROW(loader.load_gguf_model(""), std::invalid_argument);
    
    // Test with directory instead of file
    EXPECT_THROW(loader.load_gguf_model("."), std::runtime_error);
}

TEST_F(ErrorHandlingTest, Memory_Allocation_Limits) {
    // Test extremely large tensor creation
    // Note: This might not throw on all systems, but should be handled gracefully
    try {
        TensorShape huge_shape({SIZE_MAX / 8, SIZE_MAX / 8}); // Very large
        // This should either succeed or throw, but not crash
    } catch (const std::exception& e) {
        // Expected behavior for systems that can't allocate this much
        SUCCEED();
    }
}

TEST_F(ErrorHandlingTest, Exception_Safety_Guarantee) {
    // Test that operations provide basic exception safety
    TensorShape shape({10, 10});
    Tensor tensor(shape, DataType::kFloat32);
    
    // Store original state
    auto original_shape = tensor.shape();
    auto original_dtype = tensor.dtype();
    void* original_ptr = tensor.data_ptr();
    
    try {
        // Try an operation that might throw
        std::vector<std::size_t> invalid_start = {15, 15}; // Out of bounds
        std::vector<std::size_t> invalid_end = {20, 20};
        tensor.slice(invalid_start, invalid_end);
        FAIL() << "Expected exception was not thrown";
    } catch (const std::exception& e) {
        // Original tensor should remain unchanged
        EXPECT_EQ(tensor.shape(), original_shape);
        EXPECT_EQ(tensor.dtype(), original_dtype);
        EXPECT_EQ(tensor.data_ptr(), original_ptr);
    }
}
