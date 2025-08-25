/**
 * @file tensor_engine.hpp
 * @brief Defines the TensorEngine class for managing tensor operations in TurboInfer.
 * @author J.J.G. Pleunes
 */

#pragma once

#include "tensor.hpp"
#include <memory>
#include <string>
#include <vector>
#include <cstddef>

namespace turboinfer {
namespace core {

/**
 * @enum ComputeDevice
 * @brief Supported compute devices for tensor operations.
 */
enum class ComputeDevice {
    kCPU,    ///< Central Processing Unit
    kGPU,    ///< Graphics Processing Unit (CUDA/ROCm)
    kAuto    ///< Automatic device selection
};

/**
 * @class TensorEngine
 * @brief Manages tensor operations (e.g., matrix multiplication, attention) for LLM inference.
 * 
 * The TensorEngine provides CPU and GPU backends with automatic fallback to CPU if GPU
 * is unavailable. It handles optimized implementations of common operations used in
 * transformer-based language models.
 */
class TensorEngine {
public:
    /**
     * @brief Constructs a tensor engine with the specified device.
     * @param device Target compute device.
     */
    explicit TensorEngine(ComputeDevice device = ComputeDevice::kAuto);

    /**
     * @brief Destructor.
     */
    ~TensorEngine();

    /**
     * @brief Gets the currently active compute device.
     * @return Active compute device.
     */
    ComputeDevice device() const noexcept { return device_; }

    /**
     * @brief Checks if GPU acceleration is available.
     * @return True if GPU is available, false otherwise.
     */
    bool gpu_available() const noexcept;

    /**
     * @brief Gets device information as a string.
     * @return Device information string.
     */
    std::string device_info() const;

    // Matrix Operations
    
    /**
     * @brief Performs matrix multiplication: C = A * B.
     * @param a Input matrix A (M x K).
     * @param b Input matrix B (K x N).
     * @return Output matrix C (M x N).
     * @throws std::runtime_error if matrix dimensions are incompatible.
     */
    Tensor matmul(const Tensor& a, const Tensor& b);

    /**
     * @brief Performs batched matrix multiplication.
     * @param a Input tensor A (batch_size, M, K).
     * @param b Input tensor B (batch_size, K, N).
     * @return Output tensor (batch_size, M, N).
     * @throws std::runtime_error if tensor dimensions are incompatible.
     */
    Tensor batch_matmul(const Tensor& a, const Tensor& b);

    /**
     * @brief Adds a bias vector to each row of a matrix.
     * @param input Input matrix (M x N).
     * @param bias Bias vector (N).
     * @return Output matrix with bias added.
     * @throws std::runtime_error if dimensions are incompatible.
     */
    Tensor add_bias(const Tensor& input, const Tensor& bias);

    // Activation Functions

    /**
     * @brief Applies ReLU activation function.
     * @param input Input tensor.
     * @return Output tensor with ReLU applied element-wise.
     */
    Tensor relu(const Tensor& input);

    /**
     * @brief Applies GELU activation function.
     * @param input Input tensor.
     * @return Output tensor with GELU applied element-wise.
     */
    Tensor gelu(const Tensor& input);

    /**
     * @brief Applies SiLU (Swish) activation function.
     * @param input Input tensor.
     * @return Output tensor with SiLU applied element-wise.
     */
    Tensor silu(const Tensor& input);

    /**
     * @brief Applies softmax function along the last dimension.
     * @param input Input tensor.
     * @param temperature Temperature scaling factor (default: 1.0).
     * @return Output tensor with softmax applied.
     */
    Tensor softmax(const Tensor& input, float temperature = 1.0f);

    // Attention Operations

    /**
     * @brief Computes self-attention for a given input tensor.
     * @param query Query tensor (batch_size, seq_len, hidden_size).
     * @param key Key tensor (batch_size, seq_len, hidden_size).
     * @param value Value tensor (batch_size, seq_len, hidden_size).
     * @param mask Optional attention mask (batch_size, seq_len, seq_len).
     * @return Attention output tensor (batch_size, seq_len, hidden_size).
     * @throws std::runtime_error if input dimensions are invalid.
     */
    Tensor attention(const Tensor& query, const Tensor& key, const Tensor& value,
                    const Tensor* mask = nullptr);

    /**
     * @brief Computes multi-head attention.
     * @param query Query tensor (batch_size, seq_len, hidden_size).
     * @param key Key tensor (batch_size, seq_len, hidden_size).
     * @param value Value tensor (batch_size, seq_len, hidden_size).
     * @param num_heads Number of attention heads.
     * @param mask Optional attention mask.
     * @return Multi-head attention output.
     * @throws std::runtime_error if hidden_size is not divisible by num_heads.
     */
    Tensor multi_head_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                               size_t num_heads, const Tensor* mask = nullptr);

    // Normalization Operations

    /**
     * @brief Applies layer normalization.
     * @param input Input tensor.
     * @param weight Weight tensor for scaling.
     * @param bias Bias tensor for shifting.
     * @param eps Epsilon value for numerical stability (default: 1e-5).
     * @return Layer normalized output.
     */
    Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     float eps = 1e-5f);

    /**
     * @brief Applies RMS normalization.
     * @param input Input tensor.
     * @param weight Weight tensor for scaling.
     * @param eps Epsilon value for numerical stability (default: 1e-5).
     * @return RMS normalized output.
     */
    Tensor rms_norm(const Tensor& input, const Tensor& weight, float eps = 1e-5f);

    // Positional Encoding

    /**
     * @brief Applies rotary positional encoding (RoPE).
     * @param input Input tensor (batch_size, seq_len, hidden_size).
     * @param position_ids Position indices for each token.
     * @param rope_theta Base frequency for RoPE (default: 10000.0).
     * @return Tensor with rotary encoding applied.
     */
    Tensor apply_rope(const Tensor& input, const Tensor& position_ids,
                     float rope_theta = 10000.0f);

    // Element-wise Operations

    /**
     * @brief Adds two tensors element-wise.
     * @param a First input tensor.
     * @param b Second input tensor.
     * @return Element-wise sum.
     * @throws std::runtime_error if tensor shapes are incompatible.
     */
    Tensor add(const Tensor& a, const Tensor& b);

    /**
     * @brief Multiplies two tensors element-wise.
     * @param a First input tensor.
     * @param b Second input tensor.
     * @return Element-wise product.
     * @throws std::runtime_error if tensor shapes are incompatible.
     */
    Tensor multiply(const Tensor& a, const Tensor& b);

    /**
     * @brief Scales a tensor by a scalar value.
     * @param input Input tensor.
     * @param scale Scaling factor.
     * @return Scaled tensor.
     */
    Tensor scale(const Tensor& input, float scale);

    // Utility Operations

    /**
     * @brief Concatenates tensors along a specified dimension.
     * @param tensors Vector of tensors to concatenate.
     * @param dim Dimension along which to concatenate.
     * @return Concatenated tensor.
     * @throws std::runtime_error if tensors have incompatible shapes.
     */
    Tensor concatenate(const std::vector<Tensor>& tensors, size_t dim);

    /**
     * @brief Splits a tensor along a specified dimension.
     * @param input Input tensor to split.
     * @param split_sizes Sizes for each split.
     * @param dim Dimension along which to split.
     * @return Vector of split tensors.
     * @throws std::runtime_error if split sizes don't match input dimension.
     */
    std::vector<Tensor> split(const Tensor& input, const std::vector<size_t>& split_sizes,
                             size_t dim);

    /**
     * @brief Transposes a 2D tensor.
     * @param input Input tensor (must be 2D).
     * @return Transposed tensor.
     * @throws std::runtime_error if input is not 2D.
     */
    Tensor transpose(const Tensor& input);

    /**
     * @brief Permutes tensor dimensions.
     * @param input Input tensor.
     * @param dims New dimension order.
     * @return Permuted tensor.
     * @throws std::runtime_error if dimension order is invalid.
     */
    Tensor permute(const Tensor& input, const std::vector<size_t>& dims);

private:
    ComputeDevice device_;              ///< Active compute device
    std::unique_ptr<class TensorEngineImpl> impl_;  ///< Implementation details

    /**
     * @brief Initializes the tensor engine for the specified device.
     * @param device Target compute device.
     */
    void initialize(ComputeDevice device);

    /**
     * @brief Validates tensor compatibility for binary operations.
     * @param a First tensor.
     * @param b Second tensor.
     * @throws std::runtime_error if tensors are incompatible.
     */
    void validate_binary_op_compatibility(const Tensor& a, const Tensor& b) const;
};

/**
 * @brief Converts a compute device enum to string representation.
 * @param device Compute device.
 * @return String representation of the device.
 */
const char* device_to_string(ComputeDevice device);

} // namespace core
} // namespace turboinfer
