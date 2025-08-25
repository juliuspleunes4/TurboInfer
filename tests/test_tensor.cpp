/**
 * @file test_tensor.cpp
 * @brief Unit tests for the Tensor class.
 * @author J.J.G. Pleunes
 */

#include <gtest/gtest.h>
#include "turboinfer/core/tensor.hpp"
#include <vector>

using namespace turboinfer::core;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for tensor tests
    }

    void TearDown() override {
        // Common cleanup
    }
};

TEST_F(TensorTest, TensorShape_Construction) {
    // Test initializer list constructor
    TensorShape shape1{2, 3, 4};
    EXPECT_EQ(shape1.ndim(), 3);
    EXPECT_EQ(shape1.size(0), 2);
    EXPECT_EQ(shape1.size(1), 3);
    EXPECT_EQ(shape1.size(2), 4);
    EXPECT_EQ(shape1.total_size(), 24);

    // Test vector constructor
    std::vector<size_t> dims = {5, 6};
    TensorShape shape2(dims);
    EXPECT_EQ(shape2.ndim(), 2);
    EXPECT_EQ(shape2.total_size(), 30);
}

TEST_F(TensorTest, TensorShape_Equality) {
    TensorShape shape1{2, 3, 4};
    TensorShape shape2{2, 3, 4};
    TensorShape shape3{2, 3, 5};

    EXPECT_TRUE(shape1 == shape2);
    EXPECT_FALSE(shape1 == shape3);
    EXPECT_FALSE(shape1 != shape2);
    EXPECT_TRUE(shape1 != shape3);
}

TEST_F(TensorTest, TensorShape_OutOfRange) {
    TensorShape shape{2, 3};
    EXPECT_THROW(shape.size(2), std::out_of_range);
    EXPECT_THROW(shape.size(10), std::out_of_range);
}

TEST_F(TensorTest, Tensor_BasicConstruction) {
    TensorShape shape{2, 3};
    Tensor tensor(shape, DataType::kFloat32);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.dtype(), DataType::kFloat32);
    EXPECT_EQ(tensor.element_size(), sizeof(float));
    EXPECT_EQ(tensor.byte_size(), 2 * 3 * sizeof(float));
    EXPECT_FALSE(tensor.empty());
    EXPECT_NE(tensor.data(), nullptr);
}

TEST_F(TensorTest, Tensor_EmptyTensor) {
    TensorShape empty_shape{0};
    Tensor tensor(empty_shape);
    
    EXPECT_TRUE(tensor.empty());
    EXPECT_EQ(tensor.byte_size(), 0);
}

TEST_F(TensorTest, Tensor_DataAccess) {
    TensorShape shape{2, 2};
    Tensor tensor(shape, DataType::kFloat32);

    // Test typed data access
    float* data_ptr = tensor.data_ptr<float>();
    EXPECT_NE(data_ptr, nullptr);

    // Fill with test data
    data_ptr[0] = 1.0f;
    data_ptr[1] = 2.0f;
    data_ptr[2] = 3.0f;
    data_ptr[3] = 4.0f;

    // Verify data
    const float* const_data_ptr = tensor.data_ptr<float>();
    EXPECT_EQ(const_data_ptr[0], 1.0f);
    EXPECT_EQ(const_data_ptr[1], 2.0f);
    EXPECT_EQ(const_data_ptr[2], 3.0f);
    EXPECT_EQ(const_data_ptr[3], 4.0f);
}

TEST_F(TensorTest, Tensor_Fill) {
    TensorShape shape{3, 3};
    Tensor tensor(shape, DataType::kFloat32);

    tensor.fill<float>(5.0f);

    const float* data = tensor.data_ptr<float>();
    for (size_t i = 0; i < shape.total_size(); ++i) {
        EXPECT_EQ(data[i], 5.0f);
    }
}

TEST_F(TensorTest, Tensor_CopyConstructor) {
    TensorShape shape{2, 2};
    Tensor original(shape, DataType::kFloat32);
    original.fill<float>(3.14f);

    Tensor copy(original);

    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.dtype(), original.dtype());
    EXPECT_NE(copy.data(), original.data()); // Different memory

    const float* orig_data = original.data_ptr<float>();
    const float* copy_data = copy.data_ptr<float>();
    
    for (size_t i = 0; i < shape.total_size(); ++i) {
        EXPECT_EQ(copy_data[i], orig_data[i]);
    }
}

TEST_F(TensorTest, Tensor_MoveConstructor) {
    TensorShape shape{2, 2};
    Tensor original(shape, DataType::kFloat32);
    original.fill<float>(2.71f);
    
    void* original_data_ptr = original.data();
    Tensor moved(std::move(original));

    EXPECT_EQ(moved.shape(), shape);
    EXPECT_EQ(moved.dtype(), DataType::kFloat32);
    EXPECT_EQ(moved.data(), original_data_ptr);
    EXPECT_TRUE(original.empty()); // Original should be empty after move
}

TEST_F(TensorTest, Tensor_Clone) {
    TensorShape shape{2, 3};
    Tensor original(shape, DataType::kFloat32);
    original.fill<float>(1.5f);

    Tensor cloned = original.clone();

    EXPECT_EQ(cloned.shape(), original.shape());
    EXPECT_EQ(cloned.dtype(), original.dtype());
    EXPECT_NE(cloned.data(), original.data());

    const float* orig_data = original.data_ptr<float>();
    const float* clone_data = cloned.data_ptr<float>();
    
    for (size_t i = 0; i < shape.total_size(); ++i) {
        EXPECT_EQ(clone_data[i], orig_data[i]);
    }
}

TEST_F(TensorTest, Tensor_Reshape) {
    TensorShape original_shape{2, 3};
    Tensor tensor(original_shape, DataType::kFloat32);
    
    // Fill with sequential data
    float* data = tensor.data_ptr<float>();
    for (size_t i = 0; i < original_shape.total_size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    TensorShape new_shape{3, 2};
    Tensor reshaped = tensor.reshape(new_shape);

    EXPECT_EQ(reshaped.shape(), new_shape);
    EXPECT_EQ(reshaped.dtype(), DataType::kFloat32);

    // Data should be preserved
    const float* reshaped_data = reshaped.data_ptr<float>();
    for (size_t i = 0; i < new_shape.total_size(); ++i) {
        EXPECT_EQ(reshaped_data[i], static_cast<float>(i));
    }
}

TEST_F(TensorTest, Tensor_ReshapeInvalidSize) {
    TensorShape shape{2, 3};
    Tensor tensor(shape, DataType::kFloat32);

    TensorShape invalid_shape{2, 4}; // Different total size
    EXPECT_THROW(tensor.reshape(invalid_shape), std::runtime_error);
}

TEST_F(TensorTest, Tensor_TypeValidation) {
    TensorShape shape{2, 2};
    Tensor tensor(shape, DataType::kFloat32);

    // Valid type access
    EXPECT_NO_THROW(tensor.data_ptr<float>());

    // Invalid type access (wrong size)
    EXPECT_THROW(tensor.data_ptr<double>(), std::runtime_error);
}

TEST_F(TensorTest, DataType_Utilities) {
    EXPECT_EQ(get_dtype_size(DataType::kFloat32), sizeof(float));
    EXPECT_EQ(get_dtype_size(DataType::kInt32), sizeof(int32_t));
    EXPECT_EQ(get_dtype_size(DataType::kInt8), sizeof(int8_t));
    EXPECT_EQ(get_dtype_size(DataType::kUInt8), sizeof(uint8_t));

    EXPECT_STREQ(dtype_to_string(DataType::kFloat32), "float32");
    EXPECT_STREQ(dtype_to_string(DataType::kInt32), "int32");
    EXPECT_STREQ(dtype_to_string(DataType::kInt8), "int8");
}
