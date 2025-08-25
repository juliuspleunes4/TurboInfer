/**
 * @file test_data_types.cpp
 * @brief Unit tests for data type system.
 * @author J.J.G. Pleunes
 */

#include <gtest/gtest.h>
#include "turboinfer/core/tensor.hpp"

using namespace turboinfer::core;

class DataTypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for data type tests
    }

    void TearDown() override {
        // Cleanup
    }
};

TEST_F(DataTypeTest, DataType_Size_Calculation) {
    // Test size calculations for different data types
    EXPECT_EQ(data_type_size(DataType::kFloat32), 4);
    EXPECT_EQ(data_type_size(DataType::kFloat16), 2);
    EXPECT_EQ(data_type_size(DataType::kInt32), 4);
    EXPECT_EQ(data_type_size(DataType::kInt16), 2);
    EXPECT_EQ(data_type_size(DataType::kInt8), 1);
    EXPECT_EQ(data_type_size(DataType::kUInt8), 1);
}

TEST_F(DataTypeTest, DataType_String_Conversion) {
    // Test string representations
    EXPECT_EQ(data_type_to_string(DataType::kFloat32), "float32");
    EXPECT_EQ(data_type_to_string(DataType::kFloat16), "float16");
    EXPECT_EQ(data_type_to_string(DataType::kInt32), "int32");
    EXPECT_EQ(data_type_to_string(DataType::kInt16), "int16");
    EXPECT_EQ(data_type_to_string(DataType::kInt8), "int8");
    EXPECT_EQ(data_type_to_string(DataType::kUInt8), "uint8");
}

TEST_F(DataTypeTest, DataType_IsFloating) {
    // Test floating point detection
    EXPECT_TRUE(is_floating_point(DataType::kFloat32));
    EXPECT_TRUE(is_floating_point(DataType::kFloat16));
    EXPECT_FALSE(is_floating_point(DataType::kInt32));
    EXPECT_FALSE(is_floating_point(DataType::kInt16));
    EXPECT_FALSE(is_floating_point(DataType::kInt8));
    EXPECT_FALSE(is_floating_point(DataType::kUInt8));
}

TEST_F(DataTypeTest, DataType_IsInteger) {
    // Test integer detection
    EXPECT_FALSE(is_integer(DataType::kFloat32));
    EXPECT_FALSE(is_integer(DataType::kFloat16));
    EXPECT_TRUE(is_integer(DataType::kInt32));
    EXPECT_TRUE(is_integer(DataType::kInt16));
    EXPECT_TRUE(is_integer(DataType::kInt8));
    EXPECT_TRUE(is_integer(DataType::kUInt8));
}

TEST_F(DataTypeTest, DataType_IsSigned) {
    // Test signed/unsigned detection
    EXPECT_TRUE(is_signed(DataType::kFloat32));
    EXPECT_TRUE(is_signed(DataType::kFloat16));
    EXPECT_TRUE(is_signed(DataType::kInt32));
    EXPECT_TRUE(is_signed(DataType::kInt16));
    EXPECT_TRUE(is_signed(DataType::kInt8));
    EXPECT_FALSE(is_signed(DataType::kUInt8));
}

TEST_F(DataTypeTest, Tensor_Creation_With_Different_Types) {
    TensorShape shape({2, 3});
    
    // Test tensor creation with different data types
    Tensor tensor_f32(shape, DataType::kFloat32);
    EXPECT_EQ(tensor_f32.dtype(), DataType::kFloat32);
    EXPECT_EQ(tensor_f32.byte_size(), 2 * 3 * 4); // 6 elements * 4 bytes
    
    Tensor tensor_f16(shape, DataType::kFloat16);
    EXPECT_EQ(tensor_f16.dtype(), DataType::kFloat16);
    EXPECT_EQ(tensor_f16.byte_size(), 2 * 3 * 2); // 6 elements * 2 bytes
    
    Tensor tensor_i32(shape, DataType::kInt32);
    EXPECT_EQ(tensor_i32.dtype(), DataType::kInt32);
    EXPECT_EQ(tensor_i32.byte_size(), 2 * 3 * 4); // 6 elements * 4 bytes
    
    Tensor tensor_i8(shape, DataType::kInt8);
    EXPECT_EQ(tensor_i8.dtype(), DataType::kInt8);
    EXPECT_EQ(tensor_i8.byte_size(), 2 * 3 * 1); // 6 elements * 1 byte
}

TEST_F(DataTypeTest, Memory_Alignment_Calculations) {
    TensorShape shape({100}); // 100 elements
    
    // Test memory calculations for different types
    Tensor tensor_f32(shape, DataType::kFloat32);
    EXPECT_EQ(tensor_f32.byte_size(), 400); // 100 * 4
    
    Tensor tensor_f16(shape, DataType::kFloat16);
    EXPECT_EQ(tensor_f16.byte_size(), 200); // 100 * 2
    
    Tensor tensor_u8(shape, DataType::kUInt8);
    EXPECT_EQ(tensor_u8.byte_size(), 100); // 100 * 1
}
