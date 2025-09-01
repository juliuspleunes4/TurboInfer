/**
 * @file tensor_engine.cpp
 * @brief Implementation of the TensorEngine class with SIMD optimizations.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/core/tensor_engine.hpp"
#include <stdexcept>
#include <cmath>

// SIMD intrinsics support
#ifdef TURBOINFER_SIMD_ENABLED
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <immintrin.h>
    #endif
    #ifdef __ARM_NEON
        #include <arm_neon.h>
    #endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Mathematical constants for neural network operations
namespace {
    constexpr float GELU_COEFF = 0.044715f;           ///< Coefficient for GELU approximation
    constexpr float ATTENTION_MASK_VALUE = -1e9f;    ///< Large negative value for attention masking
    
    // SIMD constants
    constexpr size_t SIMD_ALIGNMENT = 32;             ///< 256-bit alignment for AVX2
    constexpr size_t AVX2_FLOAT_COUNT = 8;            ///< Number of floats in AVX2 register
    constexpr size_t NEON_FLOAT_COUNT = 4;            ///< Number of floats in NEON register
}

#ifdef TURBOINFER_SIMD_ENABLED
namespace simd_utils {
    /**
     * @brief Check if memory is properly aligned for SIMD operations.
     * @param ptr Pointer to check for alignment.
     * @param alignment Required alignment in bytes.
     * @return True if pointer is aligned, false otherwise.
     */
    inline bool is_aligned(const void* ptr, size_t alignment) {
        return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
    }
    
    /**
     * @brief Vectorized addition for float arrays using AVX2.
     * @param a First input array.
     * @param b Second input array.
     * @param result Output array.
     * @param size Number of elements to process.
     */
    void avx2_add_float(const float* a, const float* b, float* result, size_t size) {
        const size_t simd_size = size & ~(AVX2_FLOAT_COUNT - 1);
        
        for (size_t i = 0; i < simd_size; i += AVX2_FLOAT_COUNT) {
            __m256 va = _mm256_load_ps(&a[i]);
            __m256 vb = _mm256_load_ps(&b[i]);
            __m256 vresult = _mm256_add_ps(va, vb);
            _mm256_store_ps(&result[i], vresult);
        }
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
    
    /**
     * @brief Vectorized multiplication for float arrays using AVX2.
     * @param a First input array.
     * @param b Second input array.
     * @param result Output array.
     * @param size Number of elements to process.
     */
    void avx2_multiply_float(const float* a, const float* b, float* result, size_t size) {
        const size_t simd_size = size & ~(AVX2_FLOAT_COUNT - 1);
        
        for (size_t i = 0; i < simd_size; i += AVX2_FLOAT_COUNT) {
            __m256 va = _mm256_load_ps(&a[i]);
            __m256 vb = _mm256_load_ps(&b[i]);
            __m256 vresult = _mm256_mul_ps(va, vb);
            _mm256_store_ps(&result[i], vresult);
        }
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
    }
    
    /**
     * @brief Vectorized ReLU activation using AVX2.
     * @param input Input array.
     * @param result Output array.
     * @param size Number of elements to process.
     */
    void avx2_relu_float(const float* input, float* result, size_t size) {
        const size_t simd_size = size & ~(AVX2_FLOAT_COUNT - 1);
        const __m256 zero = _mm256_setzero_ps();
        
        for (size_t i = 0; i < simd_size; i += AVX2_FLOAT_COUNT) {
            __m256 vinput = _mm256_load_ps(&input[i]);
            __m256 vresult = _mm256_max_ps(vinput, zero);
            _mm256_store_ps(&result[i], vresult);
        }
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = std::max(0.0f, input[i]);
        }
    }

#ifdef __ARM_NEON
    /**
     * @brief Vectorized addition for float arrays using NEON.
     * @param a First input array.
     * @param b Second input array.
     * @param result Output array.
     * @param size Number of elements to process.
     */
    void neon_add_float(const float* a, const float* b, float* result, size_t size) {
        const size_t simd_size = size & ~(NEON_FLOAT_COUNT - 1);
        
        for (size_t i = 0; i < simd_size; i += NEON_FLOAT_COUNT) {
            float32x4_t va = vld1q_f32(&a[i]);
            float32x4_t vb = vld1q_f32(&b[i]);
            float32x4_t vresult = vaddq_f32(va, vb);
            vst1q_f32(&result[i], vresult);
        }
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
    
    /**
     * @brief Vectorized ReLU activation using NEON.
     * @param input Input array.
     * @param result Output array.
     * @param size Number of elements to process.
     */
    void neon_relu_float(const float* input, float* result, size_t size) {
        const size_t simd_size = size & ~(NEON_FLOAT_COUNT - 1);
        const float32x4_t zero = vdupq_n_f32(0.0f);
        
        for (size_t i = 0; i < simd_size; i += NEON_FLOAT_COUNT) {
            float32x4_t vinput = vld1q_f32(&input[i]);
            float32x4_t vresult = vmaxq_f32(vinput, zero);
            vst1q_f32(&result[i], vresult);
        }
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = std::max(0.0f, input[i]);
        }
    }
#endif // __ARM_NEON

    /**
     * @brief SIMD-optimized GEMM (General Matrix Multiply) implementation.
     * @param a_data Pointer to matrix A data (M x K).
     * @param b_data Pointer to matrix B data (K x N).
     * @param result_data Pointer to result matrix C data (M x N).
     * @param M Number of rows in A and C.
     * @param K Number of columns in A and rows in B.
     * @param N Number of columns in B and C.
     */
    void simd_gemm_float(const float* a_data, const float* b_data, float* result_data,
                        size_t M, size_t K, size_t N) {
        // Use tiled approach with SIMD for better cache locality and vectorization
        constexpr size_t TILE_SIZE = 64;  // Tile size for cache optimization
        
        for (size_t i_tile = 0; i_tile < M; i_tile += TILE_SIZE) {
            for (size_t j_tile = 0; j_tile < N; j_tile += TILE_SIZE) {
                for (size_t k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
                    // Process tile
                    size_t i_end = std::min(i_tile + TILE_SIZE, M);
                    size_t j_end = std::min(j_tile + TILE_SIZE, N);
                    size_t k_end = std::min(k_tile + TILE_SIZE, K);
                    
                    for (size_t i = i_tile; i < i_end; ++i) {
                        for (size_t j = j_tile; j < j_end; j += AVX2_FLOAT_COUNT) {
                            // Vectorize inner loop using AVX2
                            size_t j_vec_end = std::min(j + AVX2_FLOAT_COUNT, j_end);
                            __m256 sum_vec = _mm256_setzero_ps();
                            
                            for (size_t k = k_tile; k < k_end; ++k) {
                                __m256 a_vec = _mm256_broadcast_ss(&a_data[i * K + k]);
                                
                                if (j_vec_end - j == AVX2_FLOAT_COUNT) {
                                    __m256 b_vec = _mm256_loadu_ps(&b_data[k * N + j]);
                                    sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                                } else {
                                    // Handle remainder elements
                                    for (size_t jj = j; jj < j_vec_end; ++jj) {
                                        result_data[i * N + jj] += a_data[i * K + k] * b_data[k * N + jj];
                                    }
                                    break;
                                }
                            }
                            
                            if (j_vec_end - j == AVX2_FLOAT_COUNT) {
                                if (k_tile == 0) {
                                    _mm256_storeu_ps(&result_data[i * N + j], sum_vec);
                                } else {
                                    __m256 existing = _mm256_loadu_ps(&result_data[i * N + j]);
                                    _mm256_storeu_ps(&result_data[i * N + j], _mm256_add_ps(existing, sum_vec));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

} // namespace simd_utils
#endif // TURBOINFER_SIMD_ENABLED

namespace turboinfer {
namespace core {

// Forward declaration for implementation
class TensorEngineImpl {
public:
    ComputeDevice device = ComputeDevice::kCPU;
};

TensorEngine::TensorEngine(ComputeDevice device) 
    : device_(device), impl_(std::make_unique<TensorEngineImpl>()) {
    initialize(device);
}

TensorEngine::~TensorEngine() = default;

bool TensorEngine::gpu_available() const noexcept {
    // Placeholder implementation
    return false;
}

std::string TensorEngine::device_info() const {
    return std::string("CPU (placeholder implementation)");
}

void TensorEngine::initialize(ComputeDevice device) {
    // Placeholder implementation
    if (device == ComputeDevice::kAuto) {
        device_ = ComputeDevice::kCPU;
    } else {
        device_ = device;
    }
    impl_->device = device_;
}

// Matrix multiplication implementation for tensor operations
Tensor TensorEngine::matmul(const Tensor& a, const Tensor& b) {
    // Validate inputs
    if (a.empty() || b.empty()) {
        throw std::runtime_error("Cannot perform matrix multiplication on empty tensors");
    }
    
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Tensor data types must match for matrix multiplication");
    }
    
    const auto& shape_a = a.shape();
    const auto& shape_b = b.shape();
    
    // Support 2D matrix multiplication (M x K) * (K x N) = (M x N)
    if (shape_a.ndim() != 2 || shape_b.ndim() != 2) {
        throw std::runtime_error("Matrix multiplication currently supports only 2D tensors");
    }
    
    size_t M = shape_a.size(0);  // Rows of A
    size_t K = shape_a.size(1);  // Cols of A / Rows of B
    size_t N = shape_b.size(1);  // Cols of B
    
    if (K != shape_b.size(0)) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication: (" +
                               std::to_string(M) + "x" + std::to_string(K) + ") * (" +
                               std::to_string(shape_b.size(0)) + "x" + std::to_string(N) + ")");
    }
    
    // Create output tensor
    TensorShape output_shape({M, N});
    Tensor result(output_shape, a.dtype());
    
    // Perform matrix multiplication based on data type
    if (a.dtype() == DataType::kFloat32) {
        const float* a_data = a.data_ptr<float>();
        const float* b_data = b.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        // Basic GEMM implementation: C = A * B
        // Using row-major order
#ifdef TURBOINFER_SIMD_ENABLED
        // Initialize result to zero for tiled SIMD implementation
        std::fill(result_data, result_data + M * N, 0.0f);
        
        // Use SIMD-optimized GEMM for larger matrices
        if (M >= 32 && N >= 32 && K >= 32) {
            simd_utils::simd_gemm_float(a_data, b_data, result_data, M, K, N);
        } else {
            // Use scalar implementation for small matrices
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        sum += a_data[i * K + k] * b_data[k * N + j];
                    }
                    result_data[i * N + j] = sum;
                }
            }
        }
#else
        // Scalar implementation when SIMD is disabled
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a_data[i * K + k] * b_data[k * N + j];
                }
                result_data[i * N + j] = sum;
            }
        }
#endif
    } else {
        throw std::runtime_error("Matrix multiplication currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::batch_matmul(const Tensor& a, const Tensor& b) {
    // Validate inputs
    if (a.empty() || b.empty()) {
        throw std::runtime_error("Cannot multiply empty tensors");
    }
    
    const std::vector<size_t>& a_dims = a.shape().dimensions();
    const std::vector<size_t>& b_dims = b.shape().dimensions();
    
    // Both tensors must have at least 2 dimensions for batch matmul
    if (a_dims.size() < 2 || b_dims.size() < 2) {
        throw std::runtime_error("Batch matrix multiplication requires tensors with at least 2 dimensions");
    }
    
    // Extract matrix dimensions (last two dimensions)
    size_t a_rows = a_dims[a_dims.size() - 2];
    size_t a_cols = a_dims[a_dims.size() - 1];
    size_t b_rows = b_dims[b_dims.size() - 2];
    size_t b_cols = b_dims[b_dims.size() - 1];
    
    // Check dimension compatibility for matrix multiplication
    if (a_cols != b_rows) {
        throw std::runtime_error("Incompatible dimensions for matrix multiplication");
    }
    
    // Calculate batch dimensions
    size_t a_batch_size = 1;
    size_t b_batch_size = 1;
    
    for (size_t i = 0; i < a_dims.size() - 2; ++i) {
        a_batch_size *= a_dims[i];
    }
    for (size_t i = 0; i < b_dims.size() - 2; ++i) {
        b_batch_size *= b_dims[i];
    }
    
    // For simplicity, require same batch dimensions or one of them to be 1
    size_t batch_size = std::max(a_batch_size, b_batch_size);
    if (a_batch_size != 1 && b_batch_size != 1 && a_batch_size != b_batch_size) {
        throw std::runtime_error("Batch dimensions must be compatible for batch matrix multiplication");
    }
    
    // Create result shape: batch dimensions + [a_rows, b_cols]
    std::vector<size_t> result_dims;
    if (a_batch_size >= b_batch_size) {
        for (size_t i = 0; i < a_dims.size() - 2; ++i) {
            result_dims.push_back(a_dims[i]);
        }
    } else {
        for (size_t i = 0; i < b_dims.size() - 2; ++i) {
            result_dims.push_back(b_dims[i]);
        }
    }
    result_dims.push_back(a_rows);
    result_dims.push_back(b_cols);
    
    Tensor result(TensorShape(result_dims), a.dtype());
    
    if (a.dtype() == DataType::kFloat32) {
        const float* a_data = a.data_ptr<float>();
        const float* b_data = b.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        size_t matrix_size_a = a_rows * a_cols;
        size_t matrix_size_b = b_rows * b_cols;
        size_t matrix_size_result = a_rows * b_cols;
        
        for (size_t batch = 0; batch < batch_size; ++batch) {
            size_t a_offset = (a_batch_size == 1) ? 0 : batch * matrix_size_a;
            size_t b_offset = (b_batch_size == 1) ? 0 : batch * matrix_size_b;
            size_t result_offset = batch * matrix_size_result;
            
            // Perform matrix multiplication for this batch
            for (size_t i = 0; i < a_rows; ++i) {
                for (size_t j = 0; j < b_cols; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < a_cols; ++k) {
                        sum += a_data[a_offset + i * a_cols + k] * b_data[b_offset + k * b_cols + j];
                    }
                    result_data[result_offset + i * b_cols + j] = sum;
                }
            }
        }
    } else {
        throw std::runtime_error("Batch matrix multiplication currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::add_bias(const Tensor& input, const Tensor& bias) {
    // Validate inputs
    if (input.empty() || bias.empty()) {
        throw std::runtime_error("Cannot add bias to empty tensors");
    }
    
    const std::vector<size_t>& input_dims = input.shape().dimensions();
    const std::vector<size_t>& bias_dims = bias.shape().dimensions();
    
    // Bias should be 1D and match the last dimension of input
    if (bias_dims.size() != 1) {
        throw std::runtime_error("Bias must be a 1D tensor");
    }
    
    if (input_dims.empty() || bias_dims[0] != input_dims.back()) {
        throw std::runtime_error("Bias size must match the last dimension of input tensor");
    }
    
    // Create result tensor with same shape as input
    Tensor result(input.shape(), input.dtype());
    
    if (input.dtype() == DataType::kFloat32) {
        const float* input_data = input.data_ptr<float>();
        const float* bias_data = bias.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        size_t last_dim_size = input_dims.back();
        size_t batch_size = input.shape().total_size() / last_dim_size;
        
        for (size_t batch = 0; batch < batch_size; ++batch) {
            size_t offset = batch * last_dim_size;
            for (size_t i = 0; i < last_dim_size; ++i) {
                result_data[offset + i] = input_data[offset + i] + bias_data[i];
            }
        }
    } else {
        throw std::runtime_error("Add bias currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::relu(const Tensor& input) {
    if (input.empty()) {
        throw std::runtime_error("Cannot apply ReLU to empty tensor");
    }
    
    // Create output tensor with same shape and dtype
    Tensor result(input.shape(), input.dtype());
    
    size_t total_elements = input.shape().total_size();
    
    if (input.dtype() == DataType::kFloat32) {
        const float* input_data = input.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
#ifdef TURBOINFER_SIMD_ENABLED
        // Use SIMD optimizations if data is aligned and size is sufficient
        if (total_elements >= AVX2_FLOAT_COUNT && 
            simd_utils::is_aligned(input_data, SIMD_ALIGNMENT) &&
            simd_utils::is_aligned(result_data, SIMD_ALIGNMENT)) {
#ifdef __ARM_NEON
    simd_utils::neon_relu_float(input_data, result_data, total_elements);
#else
    simd_utils::avx2_relu_float(input_data, result_data, total_elements);
#endif
        } else {
            // Fallback to scalar implementation
            for (size_t i = 0; i < total_elements; ++i) {
                result_data[i] = std::max(0.0f, input_data[i]);
            }
        }
#else
        // Scalar implementation when SIMD is disabled
        for (size_t i = 0; i < total_elements; ++i) {
            result_data[i] = std::max(0.0f, input_data[i]);
        }
#endif
    } else {
        throw std::runtime_error("ReLU currently only supports Float32 data type");
    }
    
    return result;
}


Tensor TensorEngine::gelu(const Tensor& input) {
    if (input.empty()) {
        throw std::runtime_error("Cannot apply GELU to empty tensor");
    }
    
    // Create output tensor with same shape and dtype
    Tensor result(input.shape(), input.dtype());
    size_t total_elements = input.shape().total_size();
    
    if (input.dtype() == DataType::kFloat32) {
        const float* input_data = input.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + GELU_COEFF * x^3)))
        const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
        
        for (size_t i = 0; i < total_elements; ++i) {
            float x = input_data[i];
            float tanh_arg = sqrt_2_over_pi * (x + GELU_COEFF * x * x * x);
            result_data[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
        }
    } else {
        throw std::runtime_error("GELU currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::silu(const Tensor& input) {
    if (input.empty()) {
        throw std::runtime_error("Cannot apply SiLU to empty tensor");
    }
    
    // Create output tensor with same shape and dtype
    Tensor result(input.shape(), input.dtype());
    size_t total_elements = input.shape().total_size();
    
    if (input.dtype() == DataType::kFloat32) {
        const float* input_data = input.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        // SiLU (Swish): x * sigmoid(x) = x / (1 + exp(-x))
        for (size_t i = 0; i < total_elements; ++i) {
            float x = input_data[i];
            result_data[i] = x / (1.0f + std::exp(-x));
        }
    } else {
        throw std::runtime_error("SiLU currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::softmax(const Tensor& input, float temperature) {
    if (input.empty()) {
        throw std::runtime_error("Cannot apply softmax to empty tensor");
    }
    
    // Create output tensor with same shape and dtype
    Tensor result(input.shape(), input.dtype());
    
    if (input.dtype() == DataType::kFloat32) {
        const float* input_data = input.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        const std::vector<size_t>& shape = input.shape().dimensions();
        
        // Apply softmax along the last dimension
        size_t last_dim_size = shape.back();
        size_t batch_size = input.shape().total_size() / last_dim_size;
        
        for (size_t batch = 0; batch < batch_size; ++batch) {
            size_t offset = batch * last_dim_size;
            
            // Find max for numerical stability
            float max_val = input_data[offset];
            for (size_t i = 1; i < last_dim_size; ++i) {
                max_val = std::max(max_val, input_data[offset + i]);
            }
            
            // Compute exp(x/temperature - max) and sum
            float sum = 0.0f;
            for (size_t i = 0; i < last_dim_size; ++i) {
                float val = std::exp((input_data[offset + i] - max_val) / temperature);
                result_data[offset + i] = val;
                sum += val;
            }
            
            // Normalize
            for (size_t i = 0; i < last_dim_size; ++i) {
                result_data[offset + i] /= sum;
            }
        }
    } else {
        throw std::runtime_error("Softmax currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::attention(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor* mask) {
    if (query.empty() || key.empty() || value.empty()) {
        throw std::runtime_error("Cannot compute attention with empty tensors");
    }
    
    const std::vector<size_t>& q_dims = query.shape().dimensions();
    const std::vector<size_t>& k_dims = key.shape().dimensions();
    const std::vector<size_t>& v_dims = value.shape().dimensions();
    
    // Expect tensors with shape [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
    if (q_dims.size() < 2 || k_dims.size() < 2 || v_dims.size() < 2) {
        throw std::runtime_error("Query, key, and value tensors must have at least 2 dimensions");
    }
    
    // Check dimension compatibility
    size_t q_hidden = q_dims.back();
    size_t k_hidden = k_dims.back();
    size_t v_hidden = v_dims.back();
    size_t k_seq_len = k_dims[k_dims.size() - 2];
    size_t v_seq_len = v_dims[v_dims.size() - 2];
    
    if (q_hidden != k_hidden) {
        throw std::runtime_error("Query and key must have the same hidden dimension");
    }
    if (k_seq_len != v_seq_len) {
        throw std::runtime_error("Key and value must have the same sequence length");
    }
    
    if (query.dtype() != DataType::kFloat32) {
        throw std::runtime_error("Attention currently only supports Float32 data type");
    }
    
    // For now, assume 3D tensors [batch, seq_len, hidden] and transpose last two dims manually
    // This is a simplified implementation - full transpose will be in Phase 4
    if (k_dims.size() != 3 || q_dims.size() != 3) {
        throw std::runtime_error("Current attention implementation requires 3D tensors [batch, seq_len, hidden]");
    }
    
    size_t batch_size = k_dims[0];
    size_t k_seq = k_dims[1];
    size_t k_dim = k_dims[2];  // Renamed to avoid conflict
    
    // Create transposed key tensor [batch, hidden, seq_len]
    std::vector<size_t> transposed_shape = {batch_size, k_dim, k_seq};
    Tensor key_transposed(TensorShape(transposed_shape), key.dtype());
    
    if (key.dtype() == DataType::kFloat32) {
        const float* k_data = key.data_ptr<float>();
        float* kt_data = key_transposed.data_ptr<float>();
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < k_seq; ++s) {
                for (size_t h = 0; h < k_dim; ++h) {
                    size_t src_idx = b * k_seq * k_dim + s * k_dim + h;
                    size_t dst_idx = b * k_dim * k_seq + h * k_seq + s;
                    kt_data[dst_idx] = k_data[src_idx];
                }
            }
        }
    }
    
    // Compute attention scores: Q @ K^T
    Tensor scores = batch_matmul(query, key_transposed);
    
    // Scale by sqrt(d_k)
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(q_hidden));
    Tensor scaled_scores = scale(scores, scale_factor);
    
    // Apply mask if provided (add large negative value to masked positions)
    if (mask != nullptr) {
        // Simple masking: add -1e9 to positions where mask is 0
        const std::vector<size_t>& mask_dims = mask->shape().dimensions();
        if (mask_dims != scaled_scores.shape().dimensions()) {
            throw std::runtime_error("Mask dimensions must match attention scores");
        }
        
        if (mask->dtype() == DataType::kFloat32) {
            const float* mask_data = mask->data_ptr<float>();
            float* scores_data = scaled_scores.data_ptr<float>();
            size_t total_elements = scaled_scores.shape().total_size();
            
            for (size_t i = 0; i < total_elements; ++i) {
                if (mask_data[i] == 0.0f) {
                    scores_data[i] += ATTENTION_MASK_VALUE;  // Large negative value for masking
                }
            }
        }
    }
    
    // Apply softmax to get attention weights
    Tensor attention_weights = softmax(scaled_scores, 1.0f);
    
    // Apply attention weights to values: Attention @ V
    Tensor result = batch_matmul(attention_weights, value);
    
    return result;
}

Tensor TensorEngine::multi_head_attention(const Tensor& query, const Tensor& key, const Tensor& value, size_t num_heads, const Tensor* mask) {
    if (query.empty() || key.empty() || value.empty()) {
        throw std::runtime_error("Cannot compute multi-head attention with empty tensors");
    }
    
    const std::vector<size_t>& q_dims = query.shape().dimensions();
    const std::vector<size_t>& k_dims = key.shape().dimensions();
    const std::vector<size_t>& v_dims = value.shape().dimensions();
    
    // Expect 3D tensors [batch_size, seq_len, hidden_size]
    if (q_dims.size() != 3 || k_dims.size() != 3 || v_dims.size() != 3) {
        throw std::runtime_error("Multi-head attention requires 3D tensors [batch, seq_len, hidden]");
    }
    
    size_t batch_size = q_dims[0];
    size_t q_seq_len = q_dims[1];
    size_t hidden_size = q_dims[2];
    
    // Check that hidden_size is divisible by num_heads
    if (hidden_size % num_heads != 0) {
        throw std::runtime_error("Hidden size must be divisible by number of heads");
    }
    
    size_t head_dim = hidden_size / num_heads;
    
    if (query.dtype() != DataType::kFloat32) {
        throw std::runtime_error("Multi-head attention currently only supports Float32 data type");
    }
    
    // Split into multiple heads and compute attention for each head
    std::vector<Tensor> head_outputs;
    head_outputs.reserve(num_heads);
    
    for (size_t head = 0; head < num_heads; ++head) {
        // Extract head slice from query, key, value
        size_t start_idx = head * head_dim;
        size_t end_idx = start_idx + head_dim;
        
        // Create head-specific tensors by slicing the hidden dimension
        std::vector<size_t> head_shape = {batch_size, q_seq_len, head_dim};
        Tensor q_head(TensorShape(head_shape), query.dtype());
        Tensor k_head(TensorShape({batch_size, k_dims[1], head_dim}), key.dtype());
        Tensor v_head(TensorShape({batch_size, v_dims[1], head_dim}), value.dtype());
        
        // Copy head data
        const float* q_data = query.data_ptr<float>();
        const float* k_data = key.data_ptr<float>();
        const float* v_data = value.data_ptr<float>();
        float* qh_data = q_head.data_ptr<float>();
        float* kh_data = k_head.data_ptr<float>();
        float* vh_data = v_head.data_ptr<float>();
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < q_seq_len; ++s) {
                for (size_t h = 0; h < head_dim; ++h) {
                    size_t src_q_idx = b * q_seq_len * hidden_size + s * hidden_size + start_idx + h;
                    size_t dst_idx = b * q_seq_len * head_dim + s * head_dim + h;
                    qh_data[dst_idx] = q_data[src_q_idx];
                }
            }
            
            for (size_t s = 0; s < k_dims[1]; ++s) {
                for (size_t h = 0; h < head_dim; ++h) {
                    size_t src_k_idx = b * k_dims[1] * hidden_size + s * hidden_size + start_idx + h;
                    size_t dst_idx = b * k_dims[1] * head_dim + s * head_dim + h;
                    kh_data[dst_idx] = k_data[src_k_idx];
                }
            }
            
            for (size_t s = 0; s < v_dims[1]; ++s) {
                for (size_t h = 0; h < head_dim; ++h) {
                    size_t src_v_idx = b * v_dims[1] * hidden_size + s * hidden_size + start_idx + h;
                    size_t dst_idx = b * v_dims[1] * head_dim + s * head_dim + h;
                    vh_data[dst_idx] = v_data[src_v_idx];
                }
            }
        }
        
        // Compute attention for this head
        Tensor head_output = attention(q_head, k_head, v_head, mask);
        head_outputs.push_back(std::move(head_output));
    }
    
    // Concatenate all head outputs along the hidden dimension
    Tensor result(TensorShape({batch_size, q_seq_len, hidden_size}), query.dtype());
    float* result_data = result.data_ptr<float>();
    
    for (size_t head = 0; head < num_heads; ++head) {
        const float* head_data = head_outputs[head].data_ptr<float>();
        size_t start_idx = head * head_dim;
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < q_seq_len; ++s) {
                for (size_t h = 0; h < head_dim; ++h) {
                    size_t src_idx = b * q_seq_len * head_dim + s * head_dim + h;
                    size_t dst_idx = b * q_seq_len * hidden_size + s * hidden_size + start_idx + h;
                    result_data[dst_idx] = head_data[src_idx];
                }
            }
        }
    }
    
    return result;
}

Tensor TensorEngine::layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps) {
    if (input.empty() || weight.empty() || bias.empty()) {
        throw std::runtime_error("Cannot apply layer normalization to empty tensors");
    }
    
    const std::vector<size_t>& input_dims = input.shape().dimensions();
    const std::vector<size_t>& weight_dims = weight.shape().dimensions();
    const std::vector<size_t>& bias_dims = bias.shape().dimensions();
    
    // Weight and bias should be 1D and match the last dimension of input
    if (weight_dims.size() != 1 || bias_dims.size() != 1) {
        throw std::runtime_error("Weight and bias must be 1D tensors for layer normalization");
    }
    
    if (input_dims.empty() || weight_dims[0] != input_dims.back() || bias_dims[0] != input_dims.back()) {
        throw std::runtime_error("Weight and bias size must match the last dimension of input tensor");
    }
    
    // Create result tensor with same shape as input
    Tensor result(input.shape(), input.dtype());
    
    if (input.dtype() == DataType::kFloat32) {
        const float* input_data = input.data_ptr<float>();
        const float* weight_data = weight.data_ptr<float>();
        const float* bias_data = bias.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        size_t feature_size = input_dims.back();
        size_t batch_size = input.shape().total_size() / feature_size;
        
        for (size_t batch = 0; batch < batch_size; ++batch) {
            size_t offset = batch * feature_size;
            
            // Calculate mean
            float mean = 0.0f;
            for (size_t i = 0; i < feature_size; ++i) {
                mean += input_data[offset + i];
            }
            mean /= static_cast<float>(feature_size);
            
            // Calculate variance
            float variance = 0.0f;
            for (size_t i = 0; i < feature_size; ++i) {
                float diff = input_data[offset + i] - mean;
                variance += diff * diff;
            }
            variance /= static_cast<float>(feature_size);
            
            // Apply normalization: (x - mean) / sqrt(variance + eps) * weight + bias
            float inv_std = 1.0f / std::sqrt(variance + eps);
            for (size_t i = 0; i < feature_size; ++i) {
                float normalized = (input_data[offset + i] - mean) * inv_std;
                result_data[offset + i] = normalized * weight_data[i] + bias_data[i];
            }
        }
    } else {
        throw std::runtime_error("Layer normalization currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::rms_norm(const Tensor& input, const Tensor& weight, float eps) {
    if (input.empty() || weight.empty()) {
        throw std::runtime_error("Cannot apply RMS normalization to empty tensors");
    }
    
    const std::vector<size_t>& input_dims = input.shape().dimensions();
    const std::vector<size_t>& weight_dims = weight.shape().dimensions();
    
    // Weight should be 1D and match the last dimension of input
    if (weight_dims.size() != 1) {
        throw std::runtime_error("Weight must be a 1D tensor for RMS normalization");
    }
    
    if (input_dims.empty() || weight_dims[0] != input_dims.back()) {
        throw std::runtime_error("Weight size must match the last dimension of input tensor");
    }
    
    // Create result tensor with same shape as input
    Tensor result(input.shape(), input.dtype());
    
    if (input.dtype() == DataType::kFloat32) {
        const float* input_data = input.data_ptr<float>();
        const float* weight_data = weight.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        size_t feature_size = input_dims.back();
        size_t batch_size = input.shape().total_size() / feature_size;
        
        for (size_t batch = 0; batch < batch_size; ++batch) {
            size_t offset = batch * feature_size;
            
            // Calculate root mean square
            float sum_squares = 0.0f;
            for (size_t i = 0; i < feature_size; ++i) {
                float val = input_data[offset + i];
                sum_squares += val * val;
            }
            float rms = std::sqrt(sum_squares / static_cast<float>(feature_size) + eps);
            
            // Apply normalization: x / rms * weight
            for (size_t i = 0; i < feature_size; ++i) {
                result_data[offset + i] = (input_data[offset + i] / rms) * weight_data[i];
            }
        }
    } else {
        throw std::runtime_error("RMS normalization currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::apply_rope(const Tensor& input, const Tensor& position_ids, float rope_theta) {
    if (input.empty() || position_ids.empty()) {
        throw std::runtime_error("Cannot apply RoPE to empty tensors");
    }
    
    const std::vector<size_t>& input_dims = input.shape().dimensions();
    const std::vector<size_t>& pos_dims = position_ids.shape().dimensions();
    
    // Input should be [batch, seq_len, hidden] or [batch, num_heads, seq_len, head_dim]
    if (input_dims.size() < 3) {
        throw std::runtime_error("RoPE requires input tensor with at least 3 dimensions");
    }
    
    // Position IDs should be [batch, seq_len] or [seq_len]
    if (pos_dims.size() < 1 || pos_dims.size() > 2) {
        throw std::runtime_error("Position IDs must be 1D or 2D tensor");
    }
    
    if (input.dtype() != DataType::kFloat32 || position_ids.dtype() != DataType::kFloat32) {
        throw std::runtime_error("RoPE currently only supports Float32 data type");
    }
    
    // Create result tensor with same shape as input
    Tensor result(input.shape(), input.dtype());
    
    const float* input_data = input.data_ptr<float>();
    const float* pos_data = position_ids.data_ptr<float>();
    float* result_data = result.data_ptr<float>();
    
    size_t batch_size = input_dims[0];
    size_t seq_len, hidden_dim;
    
    // Handle both 3D [batch, seq, hidden] and 4D [batch, heads, seq, head_dim] cases
    if (input_dims.size() == 3) {
        seq_len = input_dims[1];
        hidden_dim = input_dims[2];
    } else if (input_dims.size() == 4) {
        seq_len = input_dims[2];
        hidden_dim = input_dims[3];
    } else {
        throw std::runtime_error("RoPE supports 3D or 4D input tensors only");
    }
    
    // Hidden dimension must be even for RoPE (pairs of dimensions are rotated)
    if (hidden_dim % 2 != 0) {
        throw std::runtime_error("Hidden dimension must be even for RoPE");
    }
    
    // Pre-compute frequency components
    std::vector<float> freqs;
    freqs.reserve(hidden_dim / 2);
    
    for (size_t i = 0; i < hidden_dim / 2; ++i) {
        float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / static_cast<float>(hidden_dim));
        freqs.push_back(freq);
    }
    
    // Apply RoPE rotation
    if (input_dims.size() == 3) {
        // 3D case: [batch, seq_len, hidden]
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                // Get position for this sequence element
                float position = (pos_dims.size() == 2) ? pos_data[b * seq_len + s] : pos_data[s];
                
                for (size_t i = 0; i < hidden_dim / 2; ++i) {
                    size_t idx_even = b * seq_len * hidden_dim + s * hidden_dim + 2 * i;
                    size_t idx_odd = idx_even + 1;
                    
                    float x = input_data[idx_even];
                    float y = input_data[idx_odd];
                    
                    float freq = freqs[i];
                    float cos_val = std::cos(position * freq);
                    float sin_val = std::sin(position * freq);
                    
                    // Rotate the pair (x, y) by the computed angle
                    result_data[idx_even] = x * cos_val - y * sin_val;
                    result_data[idx_odd] = x * sin_val + y * cos_val;
                }
            }
        }
    } else {
        // 4D case: [batch, num_heads, seq_len, head_dim]
        size_t num_heads = input_dims[1];
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t s = 0; s < seq_len; ++s) {
                    // Get position for this sequence element
                    float position = (pos_dims.size() == 2) ? pos_data[b * seq_len + s] : pos_data[s];
                    
                    for (size_t i = 0; i < hidden_dim / 2; ++i) {
                        size_t base_idx = b * num_heads * seq_len * hidden_dim + h * seq_len * hidden_dim + s * hidden_dim;
                        size_t idx_even = base_idx + 2 * i;
                        size_t idx_odd = idx_even + 1;
                        
                        float x = input_data[idx_even];
                        float y = input_data[idx_odd];
                        
                        float freq = freqs[i];
                        float cos_val = std::cos(position * freq);
                        float sin_val = std::sin(position * freq);
                        
                        // Rotate the pair (x, y) by the computed angle
                        result_data[idx_even] = x * cos_val - y * sin_val;
                        result_data[idx_odd] = x * sin_val + y * cos_val;
                    }
                }
            }
        }
    }
    
    return result;
}

Tensor TensorEngine::add(const Tensor& a, const Tensor& b) {
    // Validate inputs
    if (a.empty() || b.empty()) {
        throw std::runtime_error("Cannot add empty tensors");
    }
    
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Tensor data types must match for addition");
    }
    
    // Check shape compatibility (must be identical for element-wise addition)
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must match for element-wise addition");
    }
    
    // Create output tensor
    Tensor result(a.shape(), a.dtype());
    size_t total_elements = a.shape().total_size();
    
    if (a.dtype() == DataType::kFloat32) {
        const float* a_data = a.data_ptr<float>();
        const float* b_data = b.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
#ifdef TURBOINFER_SIMD_ENABLED
        // Use SIMD optimizations if data is aligned and size is sufficient
        if (total_elements >= AVX2_FLOAT_COUNT && 
            simd_utils::is_aligned(a_data, SIMD_ALIGNMENT) &&
            simd_utils::is_aligned(b_data, SIMD_ALIGNMENT) &&
            simd_utils::is_aligned(result_data, SIMD_ALIGNMENT)) {
#ifdef __ARM_NEON
            simd_utils::neon_add_float(a_data, b_data, result_data, total_elements);
#else
            simd_utils::avx2_add_float(a_data, b_data, result_data, total_elements);
#endif
        } else {
            // Fallback to scalar implementation
            for (size_t i = 0; i < total_elements; ++i) {
                result_data[i] = a_data[i] + b_data[i];
            }
        }
#else
        // Scalar implementation when SIMD is disabled
        for (size_t i = 0; i < total_elements; ++i) {
            result_data[i] = a_data[i] + b_data[i];
        }
#endif
    } else {
        throw std::runtime_error("Element-wise addition currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::multiply(const Tensor& a, const Tensor& b) {
    // Validate inputs
    if (a.empty() || b.empty()) {
        throw std::runtime_error("Cannot multiply empty tensors");
    }
    
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Tensor data types must match for multiplication");
    }
    
    // Check shape compatibility (must be identical for element-wise multiplication)
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must match for element-wise multiplication");
    }
    
    // Create output tensor
    Tensor result(a.shape(), a.dtype());
    size_t total_elements = a.shape().total_size();
    
    if (a.dtype() == DataType::kFloat32) {
        const float* a_data = a.data_ptr<float>();
        const float* b_data = b.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
#ifdef TURBOINFER_SIMD_ENABLED
        // Use SIMD optimizations if data is aligned and size is sufficient
        if (total_elements >= AVX2_FLOAT_COUNT && 
            simd_utils::is_aligned(a_data, SIMD_ALIGNMENT) &&
            simd_utils::is_aligned(b_data, SIMD_ALIGNMENT) &&
            simd_utils::is_aligned(result_data, SIMD_ALIGNMENT)) {
#ifdef __ARM_NEON
            // NEON multiplication
            const size_t simd_size = total_elements & ~(NEON_FLOAT_COUNT - 1);
            for (size_t i = 0; i < simd_size; i += NEON_FLOAT_COUNT) {
                float32x4_t va = vld1q_f32(&a_data[i]);
                float32x4_t vb = vld1q_f32(&b_data[i]);
                float32x4_t vresult = vmulq_f32(va, vb);
                vst1q_f32(&result_data[i], vresult);
            }
            // Handle remaining elements
            for (size_t i = simd_size; i < total_elements; ++i) {
                result_data[i] = a_data[i] * b_data[i];
            }
#else
            simd_utils::avx2_multiply_float(a_data, b_data, result_data, total_elements);
#endif
        } else {
            // Fallback to scalar implementation
            for (size_t i = 0; i < total_elements; ++i) {
                result_data[i] = a_data[i] * b_data[i];
            }
        }
#else
        // Scalar implementation when SIMD is disabled
        for (size_t i = 0; i < total_elements; ++i) {
            result_data[i] = a_data[i] * b_data[i];
        }
#endif
    } else {
        throw std::runtime_error("Element-wise multiplication currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::scale(const Tensor& input, float scale) {
    if (input.empty()) {
        throw std::runtime_error("Cannot scale empty tensor");
    }
    
    // Create output tensor with same shape and dtype
    Tensor result(input.shape(), input.dtype());
    size_t total_elements = input.shape().total_size();
    
    if (input.dtype() == DataType::kFloat32) {
        const float* input_data = input.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        for (size_t i = 0; i < total_elements; ++i) {
            result_data[i] = input_data[i] * scale;
        }
    } else {
        throw std::runtime_error("Scalar scaling currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::concatenate(const std::vector<Tensor>& tensors, size_t dim) {
    throw std::runtime_error("TensorEngine operations not yet implemented");
}

std::vector<Tensor> TensorEngine::split(const Tensor& input, const std::vector<size_t>& split_sizes, size_t dim) {
    throw std::runtime_error("TensorEngine operations not yet implemented");
}

Tensor TensorEngine::transpose(const Tensor& input) {
    throw std::runtime_error("TensorEngine operations not yet implemented");
}

Tensor TensorEngine::permute(const Tensor& input, const std::vector<size_t>& dims) {
    throw std::runtime_error("TensorEngine operations not yet implemented");
}

void TensorEngine::validate_binary_op_compatibility(const Tensor& a, const Tensor& b) const {
    // Basic compatibility check
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Tensor data types must match for binary operations");
    }
}

const char* device_to_string(ComputeDevice device) {
    switch (device) {
        case ComputeDevice::kCPU: return "CPU";
        case ComputeDevice::kGPU: return "GPU";
        case ComputeDevice::kAuto: return "Auto";
        default: return "Unknown";
    }
}

} // namespace core
} // namespace turboinfer
