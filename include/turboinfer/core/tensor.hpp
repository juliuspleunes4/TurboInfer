/**
 * @file tensor.hpp
 * @brief Defines the Tensor class for managing multi-dimensional arrays in TurboInfer.
 * @author J.J.G. Pleunes
 */

#pragma once

#include <vector>
#include <memory>
#include <initializer_list>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace turboinfer {
namespace core {

/**
 * @enum DataType
 * @brief Supported data types for tensor elements.
 */
enum class DataType {
    kFloat32,  ///< 32-bit floating point
    kFloat16,  ///< 16-bit floating point
    kInt32,    ///< 32-bit signed integer
    kInt16,    ///< 16-bit signed integer
    kInt8,     ///< 8-bit signed integer
    kUInt8     ///< 8-bit unsigned integer
};

/**
 * @class TensorShape
 * @brief Represents the shape (dimensions) of a tensor.
 */
class TensorShape {
public:
    /**
     * @brief Constructs a tensor shape from dimensions.
     * @param dimensions List of dimension sizes.
     */
    explicit TensorShape(std::initializer_list<size_t> dimensions);

    /**
     * @brief Constructs a tensor shape from a vector of dimensions.
     * @param dimensions Vector of dimension sizes.
     */
    explicit TensorShape(const std::vector<size_t>& dimensions);

    /**
     * @brief Gets the number of dimensions.
     * @return Number of dimensions.
     */
    size_t ndim() const noexcept { return dimensions_.size(); }

    /**
     * @brief Gets the size of a specific dimension.
     * @param dim Dimension index.
     * @return Size of the dimension.
     * @throws std::out_of_range if dimension index is invalid.
     */
    size_t size(size_t dim) const;

    /**
     * @brief Gets the total number of elements.
     * @return Total number of elements in the tensor.
     */
    size_t total_size() const noexcept { return total_size_; }

    /**
     * @brief Gets all dimensions as a vector.
     * @return Vector containing all dimension sizes.
     */
    const std::vector<size_t>& dimensions() const noexcept { return dimensions_; }

    /**
     * @brief Checks if two shapes are equal.
     * @param other Shape to compare with.
     * @return True if shapes are equal, false otherwise.
     */
    bool operator==(const TensorShape& other) const noexcept;

    /**
     * @brief Checks if two shapes are not equal.
     * @param other Shape to compare with.
     * @return True if shapes are not equal, false otherwise.
     */
    bool operator!=(const TensorShape& other) const noexcept;

private:
    std::vector<size_t> dimensions_;  ///< Dimension sizes
    size_t total_size_;               ///< Cached total size

    /**
     * @brief Calculates the total size from dimensions.
     */
    void calculate_total_size();
};

/**
 * @class Tensor
 * @brief Multi-dimensional array with support for various data types and operations.
 * 
 * The Tensor class provides efficient storage and manipulation of multi-dimensional
 * numerical data. It supports various data types, automatic memory management,
 * and optimized operations for machine learning workloads.
 */
class Tensor {
public:
    /**
     * @brief Default constructor creates an empty tensor.
     */
    Tensor() = default;

    /**
     * @brief Constructs a tensor with the specified shape and data type.
     * @param shape Shape of the tensor.
     * @param dtype Data type of tensor elements.
     */
    Tensor(const TensorShape& shape, DataType dtype = DataType::kFloat32);

    /**
     * @brief Constructs a tensor from existing data.
     * @param shape Shape of the tensor.
     * @param data Pointer to existing data (will be copied).
     * @param dtype Data type of tensor elements.
     */
    Tensor(const TensorShape& shape, const void* data, DataType dtype = DataType::kFloat32);

    /**
     * @brief Copy constructor.
     * @param other Tensor to copy from.
     */
    Tensor(const Tensor& other);

    /**
     * @brief Move constructor.
     * @param other Tensor to move from.
     */
    Tensor(Tensor&& other) noexcept;

    /**
     * @brief Copy assignment operator.
     * @param other Tensor to copy from.
     * @return Reference to this tensor.
     */
    Tensor& operator=(const Tensor& other);

    /**
     * @brief Move assignment operator.
     * @param other Tensor to move from.
     * @return Reference to this tensor.
     */
    Tensor& operator=(Tensor&& other) noexcept;

    /**
     * @brief Destructor.
     */
    ~Tensor() = default;

    /**
     * @brief Gets the shape of the tensor.
     * @return Tensor shape.
     */
    const TensorShape& shape() const noexcept { return shape_; }

    /**
     * @brief Gets the data type of tensor elements.
     * @return Data type.
     */
    DataType dtype() const noexcept { return dtype_; }

    /**
     * @brief Gets the size of a single element in bytes.
     * @return Element size in bytes.
     */
    size_t element_size() const noexcept;

    /**
     * @brief Gets the total size of tensor data in bytes.
     * @return Total data size in bytes.
     */
    size_t byte_size() const noexcept;

    /**
     * @brief Gets a pointer to the raw data.
     * @return Pointer to tensor data.
     */
    void* data() noexcept { return data_.get(); }

    /**
     * @brief Gets a const pointer to the raw data.
     * @return Const pointer to tensor data.
     */
    const void* data() const noexcept { return data_.get(); }

    /**
     * @brief Gets a typed pointer to the data.
     * @tparam T Element type.
     * @return Typed pointer to tensor data.
     * @throws std::runtime_error if type doesn't match tensor data type.
     */
    template<typename T>
    T* data_ptr();

    /**
     * @brief Gets a const typed pointer to the data.
     * @tparam T Element type.
     * @return Const typed pointer to tensor data.
     * @throws std::runtime_error if type doesn't match tensor data type.
     */
    template<typename T>
    const T* data_ptr() const;

    /**
     * @brief Checks if the tensor is empty.
     * @return True if tensor has no data, false otherwise.
     */
    bool empty() const noexcept { return !data_ || shape_.total_size() == 0; }

    /**
     * @brief Fills the tensor with a scalar value.
     * @tparam T Value type.
     * @param value Value to fill with.
     */
    template<typename T>
    void fill(T value);

    /**
     * @brief Creates a copy of this tensor.
     * @return Copy of the tensor.
     */
    Tensor clone() const;

    /**
     * @brief Reshapes the tensor to a new shape.
     * @param new_shape New shape for the tensor.
     * @return Reshaped tensor view.
     * @throws std::runtime_error if new shape is incompatible.
     */
    Tensor reshape(const TensorShape& new_shape) const;

    /**
     * @brief Creates a slice of the tensor.
     * @param start Starting indices for each dimension.
     * @param end Ending indices for each dimension.
     * @return Tensor slice.
     * @throws std::runtime_error if indices are invalid.
     */
    Tensor slice(const std::vector<size_t>& start, const std::vector<size_t>& end) const;

private:
    TensorShape shape_;                     ///< Shape of the tensor
    DataType dtype_ = DataType::kFloat32;   ///< Data type of elements
    std::unique_ptr<uint8_t[]> data_;       ///< Raw data storage

    /**
     * @brief Allocates memory for tensor data.
     */
    void allocate_memory();

    /**
     * @brief Validates that a type matches the tensor's data type.
     * @tparam T Type to validate.
     * @throws std::runtime_error if type doesn't match.
     */
    template<typename T>
    void validate_type() const;
};

/**
 * @brief Gets the size in bytes for a given data type.
 * @param dtype Data type.
 * @return Size in bytes.
 */
size_t get_dtype_size(DataType dtype);

/**
 * @brief Converts a data type to a string representation.
 * @param dtype Data type.
 * @return String representation of the data type.
 */
const char* dtype_to_string(DataType dtype);

} // namespace core
} // namespace turboinfer
