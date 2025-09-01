/**
 * @file tensor.cpp
 * @brief Implementation of the Tensor class for managing multi-dimensional arrays.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/core/tensor.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <string>
#include <vector>
#include <cstddef>

namespace turboinfer {
namespace core {

// TensorShape implementation

TensorShape::TensorShape(std::initializer_list<size_t> dimensions)
    : dimensions_(dimensions) {
    calculate_total_size();
}

TensorShape::TensorShape(const std::vector<size_t>& dimensions)
    : dimensions_(dimensions) {
    calculate_total_size();
}

size_t TensorShape::size(size_t dim) const {
    if (dim >= dimensions_.size()) {
        throw std::out_of_range("Dimension index " + std::to_string(dim) + 
                               " is out of range for tensor with " + 
                               std::to_string(dimensions_.size()) + " dimensions");
    }
    return dimensions_[dim];
}

bool TensorShape::operator==(const TensorShape& other) const noexcept {
    return dimensions_ == other.dimensions_;
}

bool TensorShape::operator!=(const TensorShape& other) const noexcept {
    return !(*this == other);
}

void TensorShape::calculate_total_size() {
    if (dimensions_.empty()) {
        total_size_ = 0;
    } else {
        total_size_ = std::accumulate(dimensions_.begin(), dimensions_.end(), 
                                     size_t(1), std::multiplies<size_t>());
    }
}

// Tensor implementation

Tensor::Tensor(const TensorShape& shape, DataType dtype)
    : shape_(shape), dtype_(dtype) {
    allocate_memory();
}

Tensor::Tensor(const TensorShape& shape, const void* data, DataType dtype)
    : shape_(shape), dtype_(dtype) {
    allocate_memory();
    if (data && !empty()) {
        std::memcpy(data_.get(), data, byte_size());
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_) {
    allocate_memory();
    if (other.data_ && !empty()) {
        std::memcpy(data_.get(), other.data_.get(), byte_size());
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), 
      dtype_(other.dtype_),
      data_(std::move(other.data_)) {
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        allocate_memory();
        if (other.data_ && !empty()) {
            std::memcpy(data_.get(), other.data_.get(), byte_size());
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        data_ = std::move(other.data_);
    }
    return *this;
}

size_t Tensor::element_size() const noexcept {
    return get_dtype_size(dtype_);
}

size_t Tensor::byte_size() const noexcept {
    return shape_.total_size() * element_size();
}

template<typename T>
T* Tensor::data_ptr() {
    validate_type<T>();
    return reinterpret_cast<T*>(data_.get());
}

template<typename T>
const T* Tensor::data_ptr() const {
    validate_type<T>();
    return reinterpret_cast<const T*>(data_.get());
}

template<typename T>
void Tensor::fill(T value) {
    validate_type<T>();
    if (empty()) return;
    
    T* ptr = data_ptr<T>();
    std::fill(ptr, ptr + shape_.total_size(), value);
}

Tensor Tensor::clone() const {
    return Tensor(shape_, data_.get(), dtype_);
}

Tensor Tensor::reshape(const TensorShape& new_shape) const {
    if (new_shape.total_size() != shape_.total_size()) {
        throw std::runtime_error("Cannot reshape tensor: new shape total size (" + 
                                std::to_string(new_shape.total_size()) + 
                                ") must match original size (" + 
                                std::to_string(shape_.total_size()) + ")");
    }
    
    Tensor result(new_shape, dtype_);
    if (!empty()) {
        std::memcpy(result.data_.get(), data_.get(), byte_size());
    }
    return result;
}

Tensor Tensor::slice(const std::vector<size_t>& start, const std::vector<size_t>& end) const {
    if (start.size() != shape_.ndim() || end.size() != shape_.ndim()) {
        throw std::runtime_error("Slice indices must have same number of dimensions as tensor");
    }
    
    // Validate slice bounds
    for (size_t i = 0; i < shape_.ndim(); ++i) {
        if (start[i] >= shape_.size(i) || end[i] > shape_.size(i) || start[i] >= end[i]) {
            throw std::runtime_error("Invalid slice bounds for dimension " + std::to_string(i));
        }
    }
    
    // Calculate new shape
    std::vector<size_t> new_dims;
    for (size_t i = 0; i < shape_.ndim(); ++i) {
        new_dims.push_back(end[i] - start[i]);
    }
    
    TensorShape new_shape(new_dims);
    Tensor result(new_shape, dtype_);
    
    // Optimized multi-dimensional slicing implementation
    if (!empty() && !result.empty()) {
        const uint8_t* src = static_cast<const uint8_t*>(data());
        uint8_t* dst = static_cast<uint8_t*>(result.data());
        size_t elem_size = element_size();
        
        // Fast path for contiguous 1D slicing
        if (shape_.ndim() == 1) {
            size_t copy_size = (end[0] - start[0]) * elem_size;
            std::memcpy(dst, src + start[0] * elem_size, copy_size);
        }
        // Optimized 2D slicing with row-wise copy
        else if (shape_.ndim() == 2) {
            size_t src_row_size = shape_.size(1) * elem_size;
            size_t copy_size = (end[1] - start[1]) * elem_size;
            
            for (size_t row = 0; row < new_shape.size(0); ++row) {
                const uint8_t* src_row = src + (start[0] + row) * src_row_size + start[1] * elem_size;
                uint8_t* dst_row = dst + row * copy_size;
                std::memcpy(dst_row, src_row, copy_size);
            }
        }
        // Optimized 3D slicing with plane-wise copy
        else if (shape_.ndim() == 3) {
            size_t src_plane_size = shape_.size(1) * shape_.size(2) * elem_size;
            size_t src_row_size = shape_.size(2) * elem_size;
            size_t dst_row_size = (end[2] - start[2]) * elem_size;
            size_t copy_size = dst_row_size;
            
            for (size_t plane = 0; plane < new_shape.size(0); ++plane) {
                for (size_t row = 0; row < new_shape.size(1); ++row) {
                    const uint8_t* src_pos = src + (start[0] + plane) * src_plane_size 
                                                + (start[1] + row) * src_row_size 
                                                + start[2] * elem_size;
                    uint8_t* dst_pos = dst + plane * new_shape.size(1) * dst_row_size 
                                           + row * dst_row_size;
                    std::memcpy(dst_pos, src_pos, copy_size);
                }
            }
        }
        // Generic N-dimensional slicing for higher dimensions
        else {
            // Pre-calculate strides for efficient coordinate conversion
            std::vector<size_t> src_strides(shape_.ndim());
            std::vector<size_t> dst_strides(new_shape.ndim());
            
            // Calculate source strides
            src_strides[shape_.ndim() - 1] = elem_size;
            for (int i = static_cast<int>(shape_.ndim()) - 2; i >= 0; --i) {
                src_strides[i] = src_strides[i + 1] * shape_.size(i + 1);
            }
            
            // Calculate destination strides
            dst_strides[new_shape.ndim() - 1] = elem_size;
            for (int i = static_cast<int>(new_shape.ndim()) - 2; i >= 0; --i) {
                dst_strides[i] = dst_strides[i + 1] * new_shape.size(i + 1);
            }
            
            // For efficiency, try to find contiguous chunks to copy
            size_t total_elements = new_shape.total_size();
            std::vector<size_t> dst_coords(new_shape.ndim());
            
            for (size_t dst_linear = 0; dst_linear < total_elements; ++dst_linear) {
                // Convert linear index to coordinates
                size_t temp = dst_linear;
                for (size_t dim = 0; dim < new_shape.ndim(); ++dim) {
                    size_t stride_elements = dst_strides[dim] / elem_size;
                    dst_coords[dim] = temp / stride_elements;
                    temp %= stride_elements;
                }
                
                // Calculate source offset
                size_t src_offset = 0;
                for (size_t dim = 0; dim < shape_.ndim(); ++dim) {
                    src_offset += (start[dim] + dst_coords[dim]) * (src_strides[dim] / elem_size);
                }
                
                // Copy element
                std::memcpy(dst + dst_linear * elem_size, 
                           src + src_offset * elem_size, 
                           elem_size);
            }
        }
    }
    
    return result;
}

void Tensor::allocate_memory() {
    if (shape_.total_size() == 0) {
        data_.reset();
        return;
    }
    
    size_t total_bytes = byte_size();
    data_ = std::make_unique<uint8_t[]>(total_bytes);
    
    // Initialize memory to zero
    std::memset(data_.get(), 0, total_bytes);
}

template<typename T>
void Tensor::validate_type() const {
    // This is a simplified type validation
    // A complete implementation would check the actual type correspondence
    if (sizeof(T) != element_size()) {
        throw std::runtime_error("Type size mismatch: expected " + 
                                std::to_string(element_size()) + " bytes, got " + 
                                std::to_string(sizeof(T)) + " bytes");
    }
}

// Utility functions

size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat32: return sizeof(float);
        case DataType::kFloat16: return sizeof(uint16_t); // 16-bit representation
        case DataType::kInt32: return sizeof(int32_t);
        case DataType::kInt16: return sizeof(int16_t);
        case DataType::kInt8: return sizeof(int8_t);
        case DataType::kUInt8: return sizeof(uint8_t);
        default:
            throw std::runtime_error("Unknown data type");
    }
}

const char* dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat32: return "float32";
        case DataType::kFloat16: return "float16";
        case DataType::kInt32: return "int32";
        case DataType::kInt16: return "int16";
        case DataType::kInt8: return "int8";
        case DataType::kUInt8: return "uint8";
        default: return "unknown";
    }
}

// Explicit template instantiations for common types
template float* Tensor::data_ptr<float>();
template const float* Tensor::data_ptr<float>() const;
template void Tensor::fill<float>(float);

template int32_t* Tensor::data_ptr<int32_t>();
template const int32_t* Tensor::data_ptr<int32_t>() const;
template void Tensor::fill<int32_t>(int32_t);

template uint8_t* Tensor::data_ptr<uint8_t>();
template const uint8_t* Tensor::data_ptr<uint8_t>() const;
template void Tensor::fill<uint8_t>(uint8_t);

template int8_t* Tensor::data_ptr<int8_t>();
template const int8_t* Tensor::data_ptr<int8_t>() const;
template void Tensor::fill<int8_t>(int8_t);

} // namespace core
} // namespace turboinfer
