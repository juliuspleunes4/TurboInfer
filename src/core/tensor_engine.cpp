/**
 * @file tensor_engine.cpp
 * @brief Implementation of the TensorEngine class with SIMD optimizations.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/core/tensor_engine.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <sstream>
#include <thread>
#include <algorithm>

// Platform-specific headers for GPU detection
#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
    #include <sys/sysinfo.h>
#endif

// OpenMP support
#ifdef _OPENMP
    #include <omp.h>
#endif

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
        // Enhanced tiled approach with optimized cache blocking and unrolling
        constexpr size_t TILE_M = 64;    // Optimized for L1 cache
        constexpr size_t TILE_N = 128;   // Larger N tile for better vectorization  
        constexpr size_t TILE_K = 256;   // Larger K tile for better reuse
        constexpr size_t UNROLL_M = 4;   // Unroll factor for M dimension
        
        // Initialize result to zero
        std::fill(result_data, result_data + M * N, 0.0f);
        
        #pragma omp parallel for schedule(dynamic) if(M > 32)
        for (size_t i_tile = 0; i_tile < M; i_tile += TILE_M) {
            for (size_t k_tile = 0; k_tile < K; k_tile += TILE_K) {
                for (size_t j_tile = 0; j_tile < N; j_tile += TILE_N) {
                    // Process tile with optimized inner loops
                    size_t i_end = std::min(i_tile + TILE_M, M);
                    size_t k_end = std::min(k_tile + TILE_K, K);
                    size_t j_end = std::min(j_tile + TILE_N, N);
                    
                    // Process in unrolled blocks for better instruction-level parallelism
                    for (size_t i = i_tile; i < i_end; i += UNROLL_M) {
                        size_t i_unroll_end = std::min(i + UNROLL_M, i_end);
                        
                        for (size_t j = j_tile; j < j_end; j += AVX2_FLOAT_COUNT) {
                            size_t j_vec_end = std::min(j + AVX2_FLOAT_COUNT, j_end);
                            
                            if (j_vec_end - j == AVX2_FLOAT_COUNT) {
                                // Process multiple rows simultaneously (unrolled)
                                __m256 sum_vec[UNROLL_M];
                                for (size_t u = 0; u < UNROLL_M && i + u < i_unroll_end; ++u) {
                                    sum_vec[u] = _mm256_loadu_ps(&result_data[(i + u) * N + j]);
                                }
                                
                                for (size_t k = k_tile; k < k_end; ++k) {
                                    __m256 b_vec = _mm256_loadu_ps(&b_data[k * N + j]);
                                    
                                    for (size_t u = 0; u < UNROLL_M && i + u < i_unroll_end; ++u) {
                                        __m256 a_vec = _mm256_broadcast_ss(&a_data[(i + u) * K + k]);
                                        sum_vec[u] = _mm256_fmadd_ps(a_vec, b_vec, sum_vec[u]);
                                    }
                                }
                                
                                // Store results
                                for (size_t u = 0; u < UNROLL_M && i + u < i_unroll_end; ++u) {
                                    _mm256_storeu_ps(&result_data[(i + u) * N + j], sum_vec[u]);
                                }
                            } else {
                                // Handle remainder elements (scalar fallback)
                                for (size_t ii = i; ii < i_unroll_end; ++ii) {
                                    for (size_t jj = j; jj < j_vec_end; ++jj) {
                                        float sum = result_data[ii * N + jj];
                                        for (size_t k = k_tile; k < k_end; ++k) {
                                            sum += a_data[ii * K + k] * b_data[k * N + jj];
                                        }
                                        result_data[ii * N + jj] = sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    /**
     * @brief Fast approximation of exp() using AVX2 for 8 floats simultaneously.
     * @param x Input values (8 floats).
     * @return Approximate exp(x) values.
     */
    __m256 fast_exp_avx2(__m256 x) {
        // Clamp x to avoid overflow/underflow
        const __m256 max_x = _mm256_set1_ps(88.0f);
        const __m256 min_x = _mm256_set1_ps(-88.0f);
        x = _mm256_max_ps(_mm256_min_ps(x, max_x), min_x);
        
        // Use polynomial approximation: exp(x) ≈ 2^(x/ln(2))
        const __m256 log2e = _mm256_set1_ps(1.44269504f);  // 1/ln(2)
        const __m256 ln2 = _mm256_set1_ps(0.69314718f);    // ln(2)
        
        // Convert to base 2: x = x * log2(e)
        __m256 x_log2e = _mm256_mul_ps(x, log2e);
        
        // Split into integer and fractional parts
        __m256 fx = _mm256_floor_ps(x_log2e);
        __m256 x_frac = _mm256_sub_ps(x_log2e, fx);
        
        // Convert integer part to actual powers of 2
        __m256i fx_int = _mm256_cvtps_epi32(fx);
        __m256i exp_int = _mm256_slli_epi32(_mm256_add_epi32(fx_int, _mm256_set1_epi32(127)), 23);
        __m256 exp_int_float = _mm256_castsi256_ps(exp_int);
        
        // Polynomial approximation for 2^frac (frac in [0,1])
        // 2^x ≈ 1 + x*ln(2) + x²*ln²(2)/2 + x³*ln³(2)/6 + ...
        const __m256 c1 = _mm256_set1_ps(0.69314718f);    // ln(2)
        const __m256 c2 = _mm256_set1_ps(0.24022651f);    // ln²(2)/2
        const __m256 c3 = _mm256_set1_ps(0.05550410f);    // ln³(2)/6
        const __m256 c4 = _mm256_set1_ps(0.00961812f);    // ln⁴(2)/24
        
        __m256 x_frac_ln2 = _mm256_mul_ps(x_frac, ln2);
        __m256 poly = _mm256_set1_ps(1.0f);
        poly = _mm256_fmadd_ps(x_frac_ln2, c1, poly);
        poly = _mm256_fmadd_ps(_mm256_mul_ps(x_frac_ln2, x_frac_ln2), c2, poly);
        __m256 x_frac_ln2_3 = _mm256_mul_ps(_mm256_mul_ps(x_frac_ln2, x_frac_ln2), x_frac_ln2);
        poly = _mm256_fmadd_ps(x_frac_ln2_3, c3, poly);
        __m256 x_frac_ln2_4 = _mm256_mul_ps(x_frac_ln2_3, x_frac_ln2);
        poly = _mm256_fmadd_ps(x_frac_ln2_4, c4, poly);
        
        // Combine integer and fractional parts
        return _mm256_mul_ps(exp_int_float, poly);
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
    // Check for GPU availability using multiple detection methods
    #ifdef _WIN32
        // Windows: Check for CUDA/OpenCL libraries
        HMODULE cuda_dll = LoadLibraryA("nvcuda.dll");
        if (cuda_dll) {
            FreeLibrary(cuda_dll);
            return true;
        }
        
        // Check for OpenCL
        HMODULE opencl_dll = LoadLibraryA("OpenCL.dll");
        if (opencl_dll) {
            FreeLibrary(opencl_dll);
            return true;
        }
    #else
        // Linux/Unix: Check for CUDA driver
        if (access("/proc/driver/nvidia/version", F_OK) == 0) {
            return true;
        }
        
        // Check for AMD GPU
        if (access("/sys/class/drm", F_OK) == 0) {
            // Basic AMD GPU detection
            return true;
        }
    #endif
    
    return false;
}

std::string TensorEngine::device_info() const {
    std::ostringstream info;
    info << "TensorEngine Device Information:\n";
    
    // CPU Information
    info << "  CPU: ";
    #ifdef _WIN32
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        info << sysInfo.dwNumberOfProcessors << " cores";
    #else
        info << sysconf(_SC_NPROCESSORS_ONLN) << " cores";
    #endif
    
    // SIMD Support
    info << " (SIMD: ";
    #ifdef __AVX512F__
        info << "AVX-512";
    #elif defined(__AVX2__)
        info << "AVX2";
    #elif defined(__AVX__)
        info << "AVX";
    #elif defined(__SSE4_2__)
        info << "SSE4.2";
    #elif defined(__SSE2__)
        info << "SSE2";
    #else
        info << "None";
    #endif
    info << ")\n";
    
    // OpenMP Support
    #ifdef _OPENMP
        info << "  OpenMP: " << _OPENMP << " (threads: " << omp_get_max_threads() << ")\n";
    #else
        info << "  OpenMP: Not available\n";
    #endif
    
    // GPU Information
    if (gpu_available()) {
        info << "  GPU: Available";
        #ifdef _WIN32
            // Try to get more detailed GPU info
            HMODULE cuda_dll = LoadLibraryA("nvcuda.dll");
            if (cuda_dll) {
                info << " (NVIDIA CUDA capable)";
                FreeLibrary(cuda_dll);
            }
        #endif
    } else {
        info << "  GPU: Not available";
    }
    
    // Current device
    info << "\n  Active Device: ";
    switch (device_) {
        case ComputeDevice::kCPU:
            info << "CPU";
            break;
        case ComputeDevice::kGPU:
            info << "GPU";
            break;
        case ComputeDevice::kAuto:
            info << "Auto (using " << (gpu_available() ? "GPU" : "CPU") << ")";
            break;
    }
    
    return info.str();
}

void TensorEngine::initialize(ComputeDevice device) {
    // Enhanced initialization with proper device selection
    if (device == ComputeDevice::kAuto) {
        // Auto-select the best available device
        if (gpu_available()) {
            device_ = ComputeDevice::kGPU;
        } else {
            device_ = ComputeDevice::kCPU;
        }
    } else {
        // Use specified device, but validate availability
        if (device == ComputeDevice::kGPU && !gpu_available()) {
            throw std::runtime_error("GPU device requested but not available. Use ComputeDevice::kAuto or kCPU instead.");
        }
        device_ = device;
    }
    
    impl_->device = device_;
    
    // Initialize device-specific resources
    switch (device_) {
        case ComputeDevice::kCPU: {
            // Initialize CPU optimizations
            #ifdef _OPENMP
                // Set optimal thread count for CPU operations
                int optimal_threads = std::min(omp_get_max_threads(), 
                    static_cast<int>(std::thread::hardware_concurrency()));
                omp_set_num_threads(optimal_threads);
            #endif
            break;
        }
            
        case ComputeDevice::kGPU: {
            // Initialize GPU resources when available
            // This would include CUDA context initialization, memory pool setup, etc.
            // For now, we gracefully fall back to CPU if GPU init fails
            try {
                // GPU initialization code would go here
                // If it fails, fall back to CPU
            } catch (const std::exception&) {
                device_ = ComputeDevice::kCPU;
                impl_->device = device_;
            }
            break;
        }
            
        case ComputeDevice::kAuto:
            // This case is handled above
            break;
    }
    
    // Validate SIMD support and log capabilities
    #ifdef TURBOINFER_SIMD_ENABLED
        // Log SIMD capabilities for debugging
        #ifdef __AVX512F__
            // AVX-512 support detected
        #elif defined(__AVX2__)
            // AVX2 support detected
        #elif defined(__AVX__)
            // AVX support detected
        #endif
    #endif
}

// Matrix multiplication implementation for tensor operations
Tensor TensorEngine::matmul(const Tensor& a, const Tensor& b) {
    // Validate inputs
    if (a.empty() || b.empty()) {
        throw std::runtime_error("Cannot perform matrix multiplication on empty tensors");
    }
    
    // Handle mixed data types by converting to Float32 for computation
    Tensor a_converted = a;
    Tensor b_converted = b;
    
    if (a.dtype() != DataType::kFloat32) {
        a_converted = convert_dtype(a, DataType::kFloat32);
    }
    if (b.dtype() != DataType::kFloat32) {
        b_converted = convert_dtype(b, DataType::kFloat32);
    }
    
    const auto& shape_a = a_converted.shape();
    const auto& shape_b = b_converted.shape();
    
    // Support both 2D and 3D matrix multiplication
    // 2D: (M x K) * (K x N) = (M x N)
    // 3D: (B x M x K) * (K x N) = (B x M x N) [batched]
    // 3D: (B x M x K) * (B x K x N) = (B x M x N) [batched with batch dimension in both]
    
    if (shape_a.ndim() == 2 && shape_b.ndim() == 2) {
        // Standard 2D matrix multiplication
        return matmul_2d(a_converted, b_converted);
    } else if (shape_a.ndim() == 3 && shape_b.ndim() == 2) {
        // Batched 3D × 2D: (B, M, K) × (K, N) → (B, M, N)
        return matmul_3d_2d(a_converted, b_converted);
    } else if (shape_a.ndim() == 3 && shape_b.ndim() == 3) {
        // Batched 3D × 3D: (B, M, K) × (B, K, N) → (B, M, N)
        return matmul_3d_3d(a_converted, b_converted);
    } else {
        throw std::runtime_error("Matrix multiplication supports 2D and 3D tensors only. Got shapes: " +
                               std::to_string(shape_a.ndim()) + "D × " + std::to_string(shape_b.ndim()) + "D");
    }
}

Tensor TensorEngine::matmul_2d(const Tensor& a, const Tensor& b) {
    const auto& shape_a = a.shape();
    const auto& shape_b = b.shape();
    
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

Tensor TensorEngine::matmul_3d_2d(const Tensor& a, const Tensor& b) {
    const auto& shape_a = a.shape();
    const auto& shape_b = b.shape();
    
    size_t B = shape_a.size(0);  // Batch size
    size_t M = shape_a.size(1);  // Rows of A matrices
    size_t K = shape_a.size(2);  // Cols of A / Rows of B
    size_t N = shape_b.size(1);  // Cols of B
    
    if (K != shape_b.size(0)) {
        throw std::runtime_error("3D×2D Matrix dimensions incompatible: (" +
                               std::to_string(B) + "x" + std::to_string(M) + "x" + std::to_string(K) + 
                               ") * (" + std::to_string(shape_b.size(0)) + "x" + std::to_string(N) + ")");
    }
    
    // Create output tensor (B x M x N)
    TensorShape output_shape({B, M, N});
    Tensor result(output_shape, a.dtype());
    
    if (a.dtype() == DataType::kFloat32) {
        const float* a_data = a.data_ptr<float>();
        const float* b_data = b.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        // Perform batched matrix multiplication
        // For each batch, compute: A[b] * B = C[b]
        for (size_t b = 0; b < B; ++b) {
            const float* a_batch = a_data + b * M * K;  // Start of batch b in A
            float* result_batch = result_data + b * M * N;  // Start of batch b in result
            
            // Standard 2D matrix multiplication for this batch
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        sum += a_batch[i * K + k] * b_data[k * N + j];
                    }
                    result_batch[i * N + j] = sum;
                }
            }
        }
    } else {
        throw std::runtime_error("3D matrix multiplication currently only supports Float32 data type");
    }
    
    return result;
}

Tensor TensorEngine::matmul_3d_3d(const Tensor& a, const Tensor& b) {
    const auto& shape_a = a.shape();
    const auto& shape_b = b.shape();
    
    size_t B = shape_a.size(0);  // Batch size
    size_t M = shape_a.size(1);  // Rows of A matrices
    size_t K = shape_a.size(2);  // Cols of A / Rows of B
    size_t N = shape_b.size(2);  // Cols of B matrices
    
    if (B != shape_b.size(0)) {
        throw std::runtime_error("3D×3D Batch sizes must match: " +
                               std::to_string(B) + " vs " + std::to_string(shape_b.size(0)));
    }
    
    if (K != shape_b.size(1)) {
        throw std::runtime_error("3D×3D Matrix dimensions incompatible: (" +
                               std::to_string(B) + "x" + std::to_string(M) + "x" + std::to_string(K) + 
                               ") * (" + std::to_string(shape_b.size(0)) + "x" + std::to_string(shape_b.size(1)) + "x" + std::to_string(N) + ")");
    }
    
    // Create output tensor (B x M x N)
    TensorShape output_shape({B, M, N});
    Tensor result(output_shape, a.dtype());
    
    if (a.dtype() == DataType::kFloat32) {
        const float* a_data = a.data_ptr<float>();
        const float* b_data = b.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        // Perform batched matrix multiplication
        // For each batch, compute: A[b] * B[b] = C[b]
        for (size_t b = 0; b < B; ++b) {
            const float* a_batch = a_data + b * M * K;      // Start of batch b in A
            const float* b_batch = b_data + b * K * N;      // Start of batch b in B  
            float* result_batch = result_data + b * M * N;  // Start of batch b in result
            
            // Standard 2D matrix multiplication for this batch
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        sum += a_batch[i * K + k] * b_batch[k * N + j];
                    }
                    result_batch[i * N + j] = sum;
                }
            }
        }
    } else {
        throw std::runtime_error("3D matrix multiplication currently only supports Float32 data type");
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
        
        #pragma omp parallel for if(batch_size > 4)
        for (size_t batch = 0; batch < batch_size; ++batch) {
            size_t offset = batch * last_dim_size;
            
#ifdef TURBOINFER_SIMD_ENABLED
            // SIMD-optimized softmax for larger sequences
            if (last_dim_size >= AVX2_FLOAT_COUNT * 2) {
                // Find max using SIMD
                __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                size_t simd_end = (last_dim_size / AVX2_FLOAT_COUNT) * AVX2_FLOAT_COUNT;
                
                for (size_t i = 0; i < simd_end; i += AVX2_FLOAT_COUNT) {
                    __m256 data_vec = _mm256_loadu_ps(&input_data[offset + i]);
                    max_vec = _mm256_max_ps(max_vec, data_vec);
                }
                
                // Horizontal max reduction
                float max_val = -std::numeric_limits<float>::infinity();
                alignas(32) float max_array[AVX2_FLOAT_COUNT];
                _mm256_store_ps(max_array, max_vec);
                for (int i = 0; i < AVX2_FLOAT_COUNT; ++i) {
                    max_val = std::max(max_val, max_array[i]);
                }
                
                // Handle remainder elements
                for (size_t i = simd_end; i < last_dim_size; ++i) {
                    max_val = std::max(max_val, input_data[offset + i]);
                }
                
                // Compute exp and sum using SIMD
                __m256 max_broadcast = _mm256_set1_ps(max_val);
                __m256 temp_broadcast = _mm256_set1_ps(temperature);
                __m256 sum_vec = _mm256_setzero_ps();
                
                for (size_t i = 0; i < simd_end; i += AVX2_FLOAT_COUNT) {
                    __m256 data_vec = _mm256_loadu_ps(&input_data[offset + i]);
                    __m256 normalized = _mm256_div_ps(_mm256_sub_ps(data_vec, max_broadcast), temp_broadcast);
                    
                    // Fast exp approximation using SIMD
                    __m256 exp_val = simd_utils::fast_exp_avx2(normalized);
                    _mm256_storeu_ps(&result_data[offset + i], exp_val);
                    sum_vec = _mm256_add_ps(sum_vec, exp_val);
                }
                
                // Horizontal sum reduction
                float sum = 0.0f;
                alignas(32) float sum_array[AVX2_FLOAT_COUNT];
                _mm256_store_ps(sum_array, sum_vec);
                for (int i = 0; i < AVX2_FLOAT_COUNT; ++i) {
                    sum += sum_array[i];
                }
                
                // Handle remainder elements
                for (size_t i = simd_end; i < last_dim_size; ++i) {
                    float val = std::exp((input_data[offset + i] - max_val) / temperature);
                    result_data[offset + i] = val;
                    sum += val;
                }
                
                // Normalize using SIMD
                __m256 sum_broadcast = _mm256_set1_ps(sum);
                for (size_t i = 0; i < simd_end; i += AVX2_FLOAT_COUNT) {
                    __m256 data_vec = _mm256_loadu_ps(&result_data[offset + i]);
                    __m256 normalized = _mm256_div_ps(data_vec, sum_broadcast);
                    _mm256_storeu_ps(&result_data[offset + i], normalized);
                }
                
                // Handle remainder elements
                for (size_t i = simd_end; i < last_dim_size; ++i) {
                    result_data[offset + i] /= sum;
                }
            } else {
#endif
                // Scalar fallback for small sequences
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
#ifdef TURBOINFER_SIMD_ENABLED
            }
#endif
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
    
    // Fast path for small sequences (single token inference)
    if (q_dims.size() == 3 && q_dims[1] == 1 && k_dims.size() == 3) {
        // Single query token against cached keys/values
        return attention_fast_incremental(query, key, value, mask);
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

Tensor TensorEngine::attention_fast_incremental(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor* mask) {
    // Optimized attention for single query token against cached key/value tensors
    // query: [batch_size, 1, hidden_size]
    // key:   [batch_size, seq_len, hidden_size]  
    // value: [batch_size, seq_len, hidden_size]
    
    const std::vector<size_t>& q_dims = query.shape().dimensions();
    const std::vector<size_t>& k_dims = key.shape().dimensions();
    const std::vector<size_t>& v_dims = value.shape().dimensions();
    
    if (q_dims.size() != 3 || k_dims.size() != 3 || v_dims.size() != 3) {
        throw std::runtime_error("Fast incremental attention requires 3D tensors");
    }
    
    if (q_dims[1] != 1) {
        throw std::runtime_error("Fast incremental attention requires single query token (seq_len=1)");
    }
    
    size_t batch_size = q_dims[0];
    size_t hidden_size = q_dims[2];
    size_t kv_seq_len = k_dims[1];
    
    if (query.dtype() != DataType::kFloat32) {
        throw std::runtime_error("Fast incremental attention currently only supports Float32");
    }
    
    const float* q_data = query.data_ptr<float>();
    const float* k_data = key.data_ptr<float>();
    const float* v_data = value.data_ptr<float>();
    
    // Output tensor: [batch_size, 1, hidden_size]
    Tensor result(TensorShape({batch_size, 1, hidden_size}), query.dtype());
    float* result_data = result.data_ptr<float>();
    
    float scale = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    
    #pragma omp parallel for if(batch_size > 1)
    for (size_t b = 0; b < batch_size; ++b) {
        // Compute attention scores for single query against all cached keys
        std::vector<float> scores(kv_seq_len);
        
        for (size_t k_pos = 0; k_pos < kv_seq_len; ++k_pos) {
            float score = 0.0f;
            
#ifdef TURBOINFER_SIMD_ENABLED
            // SIMD dot product for query-key attention score
            size_t simd_end = (hidden_size / AVX2_FLOAT_COUNT) * AVX2_FLOAT_COUNT;
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (size_t h = 0; h < simd_end; h += AVX2_FLOAT_COUNT) {
                __m256 q_vec = _mm256_loadu_ps(&q_data[b * hidden_size + h]);
                __m256 k_vec = _mm256_loadu_ps(&k_data[b * kv_seq_len * hidden_size + k_pos * hidden_size + h]);
                sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
            }
            
            // Horizontal sum
            alignas(32) float sum_array[AVX2_FLOAT_COUNT];
            _mm256_store_ps(sum_array, sum_vec);
            for (int i = 0; i < AVX2_FLOAT_COUNT; ++i) {
                score += sum_array[i];
            }
            
            // Handle remainder
            for (size_t h = simd_end; h < hidden_size; ++h) {
                score += q_data[b * hidden_size + h] * k_data[b * kv_seq_len * hidden_size + k_pos * hidden_size + h];
            }
#else
            // Scalar dot product
            for (size_t h = 0; h < hidden_size; ++h) {
                score += q_data[b * hidden_size + h] * k_data[b * kv_seq_len * hidden_size + k_pos * hidden_size + h];
            }
#endif
            scores[k_pos] = score * scale;
        }
        
        // Apply softmax to scores
        float max_score = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        
        for (size_t k_pos = 0; k_pos < kv_seq_len; ++k_pos) {
            scores[k_pos] = std::exp(scores[k_pos] - max_score);
            sum_exp += scores[k_pos];
        }
        
        for (size_t k_pos = 0; k_pos < kv_seq_len; ++k_pos) {
            scores[k_pos] /= sum_exp;
        }
        
        // Compute weighted sum of values using attention weights
        for (size_t h = 0; h < hidden_size; ++h) {
            float output_val = 0.0f;
            
#ifdef TURBOINFER_SIMD_ENABLED
            // SIMD weighted sum
            size_t simd_kv_end = (kv_seq_len / AVX2_FLOAT_COUNT) * AVX2_FLOAT_COUNT;
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (size_t k_pos = 0; k_pos < simd_kv_end; k_pos += AVX2_FLOAT_COUNT) {
                __m256 weight_vec = _mm256_loadu_ps(&scores[k_pos]);
                __m256 value_vec = _mm256_set_ps(
                    v_data[b * kv_seq_len * hidden_size + (k_pos + 7) * hidden_size + h],
                    v_data[b * kv_seq_len * hidden_size + (k_pos + 6) * hidden_size + h],
                    v_data[b * kv_seq_len * hidden_size + (k_pos + 5) * hidden_size + h],
                    v_data[b * kv_seq_len * hidden_size + (k_pos + 4) * hidden_size + h],
                    v_data[b * kv_seq_len * hidden_size + (k_pos + 3) * hidden_size + h],
                    v_data[b * kv_seq_len * hidden_size + (k_pos + 2) * hidden_size + h],
                    v_data[b * kv_seq_len * hidden_size + (k_pos + 1) * hidden_size + h],
                    v_data[b * kv_seq_len * hidden_size + k_pos * hidden_size + h]
                );
                sum_vec = _mm256_fmadd_ps(weight_vec, value_vec, sum_vec);
            }
            
            // Horizontal sum
            alignas(32) float sum_array[AVX2_FLOAT_COUNT];
            _mm256_store_ps(sum_array, sum_vec);
            for (int i = 0; i < AVX2_FLOAT_COUNT; ++i) {
                output_val += sum_array[i];
            }
            
            // Handle remainder
            for (size_t k_pos = simd_kv_end; k_pos < kv_seq_len; ++k_pos) {
                output_val += scores[k_pos] * v_data[b * kv_seq_len * hidden_size + k_pos * hidden_size + h];
            }
#else
            // Scalar weighted sum
            for (size_t k_pos = 0; k_pos < kv_seq_len; ++k_pos) {
                output_val += scores[k_pos] * v_data[b * kv_seq_len * hidden_size + k_pos * hidden_size + h];
            }
#endif
            result_data[b * hidden_size + h] = output_val;
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
    
    // Convert inputs to Float32 if needed
    Tensor input_converted = input;
    Tensor weight_converted = weight;
    
    if (input.dtype() != DataType::kFloat32) {
        input_converted = convert_dtype(input, DataType::kFloat32);
    }
    if (weight.dtype() != DataType::kFloat32) {
        weight_converted = convert_dtype(weight, DataType::kFloat32);
    }
    
    // Create result tensor with same shape as input, but in Float32
    Tensor result(input_converted.shape(), DataType::kFloat32);
    
    const float* input_data = input_converted.data_ptr<float>();
    const float* weight_data = weight_converted.data_ptr<float>();
    float* result_data = result.data_ptr<float>();
    
    size_t feature_size = input_dims.back();
    size_t batch_size = input_converted.shape().total_size() / feature_size;
    
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
    if (tensors.empty()) {
        throw std::runtime_error("Cannot concatenate empty list of tensors");
    }
    
    if (tensors.size() == 1) {
        return tensors[0].clone();
    }
    
    // Validate all tensors have the same shape except for the concatenation dimension
    const auto& first_shape = tensors[0].shape();
    DataType dtype = tensors[0].dtype();
    
    if (dim >= first_shape.ndim()) {
        throw std::runtime_error("Concatenation dimension out of bounds");
    }
    
    size_t total_concat_size = 0;
    for (const auto& tensor : tensors) {
        if (tensor.empty()) {
            throw std::runtime_error("Cannot concatenate empty tensors");
        }
        
        if (tensor.dtype() != dtype) {
            throw std::runtime_error("All tensors must have the same data type for concatenation");
        }
        
        if (tensor.shape().ndim() != first_shape.ndim()) {
            throw std::runtime_error("All tensors must have the same number of dimensions for concatenation");
        }
        
        // Check that all dimensions match except the concatenation dimension
        for (size_t i = 0; i < first_shape.ndim(); ++i) {
            if (i != dim && tensor.shape().size(i) != first_shape.size(i)) {
                throw std::runtime_error("Tensor dimensions must match except for concatenation dimension");
            }
        }
        
        total_concat_size += tensor.shape().size(dim);
    }
    
    // Create output shape
    auto output_dims = first_shape.dimensions();
    output_dims[dim] = total_concat_size;
    TensorShape output_shape(output_dims);
    
    Tensor result(output_shape, dtype);
    
    // Perform concatenation
    if (dtype == DataType::kFloat32) {
        float* result_data = result.data_ptr<float>();
        size_t offset = 0;
        
        // Calculate strides
        size_t outer_size = 1;
        for (size_t i = 0; i < dim; ++i) {
            outer_size *= first_shape.size(i);
        }
        
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < first_shape.ndim(); ++i) {
            inner_size *= first_shape.size(i);
        }
        
        for (const auto& tensor : tensors) {
            const float* tensor_data = tensor.data_ptr<float>();
            size_t tensor_concat_size = tensor.shape().size(dim);
            size_t tensor_slice_size = tensor_concat_size * inner_size;
            
            for (size_t outer = 0; outer < outer_size; ++outer) {
                size_t src_offset = outer * tensor_slice_size;
                size_t dst_offset = outer * total_concat_size * inner_size + offset * inner_size;
                
                memcpy(result_data + dst_offset, 
                       tensor_data + src_offset, 
                       tensor_slice_size * sizeof(float));
            }
            
            offset += tensor_concat_size;
        }
    } else if (dtype == DataType::kInt32) {
        int32_t* result_data = result.data_ptr<int32_t>();
        size_t offset = 0;
        
        size_t outer_size = 1;
        for (size_t i = 0; i < dim; ++i) {
            outer_size *= first_shape.size(i);
        }
        
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < first_shape.ndim(); ++i) {
            inner_size *= first_shape.size(i);
        }
        
        for (const auto& tensor : tensors) {
            const int32_t* tensor_data = tensor.data_ptr<int32_t>();
            size_t tensor_concat_size = tensor.shape().size(dim);
            size_t tensor_slice_size = tensor_concat_size * inner_size;
            
            for (size_t outer = 0; outer < outer_size; ++outer) {
                size_t src_offset = outer * tensor_slice_size;
                size_t dst_offset = outer * total_concat_size * inner_size + offset * inner_size;
                
                memcpy(result_data + dst_offset, 
                       tensor_data + src_offset, 
                       tensor_slice_size * sizeof(int32_t));
            }
            
            offset += tensor_concat_size;
        }
    } else {
        throw std::runtime_error("Concatenation currently only supports Float32 and Int32 data types");
    }
    
    return result;
}

std::vector<Tensor> TensorEngine::split(const Tensor& input, const std::vector<size_t>& split_sizes, size_t dim) {
    if (input.empty()) {
        throw std::runtime_error("Cannot split empty tensor");
    }
    
    if (split_sizes.empty()) {
        throw std::runtime_error("Split sizes cannot be empty");
    }
    
    const auto& shape = input.shape();
    
    if (dim >= shape.ndim()) {
        throw std::runtime_error("Split dimension out of bounds");
    }
    
    // Validate split sizes sum to the dimension size
    size_t total_split_size = 0;
    for (size_t size : split_sizes) {
        if (size == 0) {
            throw std::runtime_error("Split sizes must be greater than 0");
        }
        total_split_size += size;
    }
    
    if (total_split_size != shape.size(dim)) {
        throw std::runtime_error("Sum of split sizes must equal the dimension size");
    }
    
    std::vector<Tensor> results;
    results.reserve(split_sizes.size());
    
    if (input.dtype() == DataType::kFloat32) {
        const float* input_data = input.data_ptr<float>();
        
        // Calculate strides
        size_t outer_size = 1;
        for (size_t i = 0; i < dim; ++i) {
            outer_size *= shape.size(i);
        }
        
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < shape.ndim(); ++i) {
            inner_size *= shape.size(i);
        }
        
        size_t current_offset = 0;
        
        for (size_t split_size : split_sizes) {
            // Create output shape for this split
            auto output_dims = shape.dimensions();
            output_dims[dim] = split_size;
            TensorShape output_shape(output_dims);
            
            Tensor split_tensor(output_shape, input.dtype());
            float* split_data = split_tensor.data_ptr<float>();
            
            // Copy data for this split
            size_t split_slice_size = split_size * inner_size;
            
            for (size_t outer = 0; outer < outer_size; ++outer) {
                size_t src_offset = outer * shape.size(dim) * inner_size + current_offset * inner_size;
                size_t dst_offset = outer * split_slice_size;
                
                memcpy(split_data + dst_offset, 
                       input_data + src_offset, 
                       split_slice_size * sizeof(float));
            }
            
            results.push_back(std::move(split_tensor));
            current_offset += split_size;
        }
    } else if (input.dtype() == DataType::kInt32) {
        const int32_t* input_data = input.data_ptr<int32_t>();
        
        size_t outer_size = 1;
        for (size_t i = 0; i < dim; ++i) {
            outer_size *= shape.size(i);
        }
        
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < shape.ndim(); ++i) {
            inner_size *= shape.size(i);
        }
        
        size_t current_offset = 0;
        
        for (size_t split_size : split_sizes) {
            auto output_dims = shape.dimensions();
            output_dims[dim] = split_size;
            TensorShape output_shape(output_dims);
            
            Tensor split_tensor(output_shape, input.dtype());
            int32_t* split_data = split_tensor.data_ptr<int32_t>();
            
            size_t split_slice_size = split_size * inner_size;
            
            for (size_t outer = 0; outer < outer_size; ++outer) {
                size_t src_offset = outer * shape.size(dim) * inner_size + current_offset * inner_size;
                size_t dst_offset = outer * split_slice_size;
                
                memcpy(split_data + dst_offset, 
                       input_data + src_offset, 
                       split_slice_size * sizeof(int32_t));
            }
            
            results.push_back(std::move(split_tensor));
            current_offset += split_size;
        }
    } else {
        throw std::runtime_error("Split currently only supports Float32 and Int32 data types");
    }
    
    return results;
}

Tensor TensorEngine::transpose(const Tensor& input) {
    if (input.empty()) {
        throw std::runtime_error("Cannot transpose empty tensor");
    }
    
    const auto& shape = input.shape();
    
    // Handle different tensor dimensions
    if (shape.ndim() == 1) {
        // 1D tensor: transpose is just a copy
        return input.clone();
    } else if (shape.ndim() == 2) {
        // 2D tensor: standard matrix transpose (M x N) -> (N x M)
        size_t rows = shape.size(0);
        size_t cols = shape.size(1);
        
        TensorShape transposed_shape({cols, rows});
        Tensor result(transposed_shape, input.dtype());
        
        if (input.dtype() == DataType::kFloat32) {
            const float* input_data = input.data_ptr<float>();
            float* result_data = result.data_ptr<float>();
            
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result_data[j * rows + i] = input_data[i * cols + j];
                }
            }
        } else if (input.dtype() == DataType::kInt32) {
            const int32_t* input_data = input.data_ptr<int32_t>();
            int32_t* result_data = result.data_ptr<int32_t>();
            
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result_data[j * rows + i] = input_data[i * cols + j];
                }
            }
        } else if (input.dtype() == DataType::kInt8) {
            const int8_t* input_data = input.data_ptr<int8_t>();
            int8_t* result_data = result.data_ptr<int8_t>();
            
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result_data[j * rows + i] = input_data[i * cols + j];
                }
            }
        } else if (input.dtype() == DataType::kUInt8) {
            const uint8_t* input_data = input.data_ptr<uint8_t>();
            uint8_t* result_data = result.data_ptr<uint8_t>();
            
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result_data[j * rows + i] = input_data[i * cols + j];
                }
            }
        } else {
            throw std::runtime_error("Transpose not supported for this data type");
        }
        
        return result;
    } else {
        // For higher dimensions, transpose swaps the last two dimensions
        // For example: (B, M, N) -> (B, N, M)
        auto dims = shape.dimensions();
        if (dims.size() < 2) {
            throw std::runtime_error("Cannot transpose tensor with less than 2 dimensions");
        }
        
        // Swap last two dimensions
        std::swap(dims[dims.size() - 2], dims[dims.size() - 1]);
        TensorShape transposed_shape(dims);
        Tensor result(transposed_shape, input.dtype());
        
        size_t last_dim = shape.size(shape.ndim() - 1);
        size_t second_last_dim = shape.size(shape.ndim() - 2);
        size_t batch_size = shape.total_size() / (last_dim * second_last_dim);
        
        if (input.dtype() == DataType::kFloat32) {
            const float* input_data = input.data_ptr<float>();
            float* result_data = result.data_ptr<float>();
            
            for (size_t batch = 0; batch < batch_size; ++batch) {
                size_t batch_offset = batch * second_last_dim * last_dim;
                
                for (size_t i = 0; i < second_last_dim; ++i) {
                    for (size_t j = 0; j < last_dim; ++j) {
                        size_t input_idx = batch_offset + i * last_dim + j;
                        size_t output_idx = batch_offset + j * second_last_dim + i;
                        result_data[output_idx] = input_data[input_idx];
                    }
                }
            }
        } else {
            throw std::runtime_error("Multi-dimensional transpose currently only supports Float32");
        }
        
        return result;
    }
}

Tensor TensorEngine::permute(const Tensor& input, const std::vector<size_t>& dims) {
    if (input.empty()) {
        throw std::runtime_error("Cannot permute empty tensor");
    }
    
    const auto& shape = input.shape();
    
    if (dims.size() != shape.ndim()) {
        throw std::runtime_error("Number of permutation dimensions must match tensor dimensions");
    }
    
    // Validate that dims contains each dimension exactly once
    std::vector<bool> used(shape.ndim(), false);
    for (size_t dim : dims) {
        if (dim >= shape.ndim()) {
            throw std::runtime_error("Permutation dimension out of bounds");
        }
        if (used[dim]) {
            throw std::runtime_error("Duplicate dimension in permutation");
        }
        used[dim] = true;
    }
    
    // Create permuted shape
    std::vector<size_t> permuted_dims(shape.ndim());
    for (size_t i = 0; i < dims.size(); ++i) {
        permuted_dims[i] = shape.size(dims[i]);
    }
    TensorShape permuted_shape(permuted_dims);
    
    Tensor result(permuted_shape, input.dtype());
    
    // Calculate strides for input and output
    std::vector<size_t> input_strides(shape.ndim());
    std::vector<size_t> output_strides(shape.ndim());
    
    input_strides[shape.ndim() - 1] = 1;
    for (int i = static_cast<int>(shape.ndim()) - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * shape.size(i + 1);
    }
    
    output_strides[shape.ndim() - 1] = 1;
    for (int i = static_cast<int>(shape.ndim()) - 2; i >= 0; --i) {
        output_strides[i] = output_strides[i + 1] * permuted_dims[i + 1];
    }
    
    if (input.dtype() == DataType::kFloat32) {
        const float* input_data = input.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        
        // Iterate through all elements in the output tensor
        size_t total_elements = permuted_shape.total_size();
        
        for (size_t output_idx = 0; output_idx < total_elements; ++output_idx) {
            // Convert linear output index to multi-dimensional coordinates
            std::vector<size_t> output_coords(shape.ndim());
            size_t temp_idx = output_idx;
            
            for (size_t i = 0; i < shape.ndim(); ++i) {
                output_coords[i] = temp_idx / output_strides[i];
                temp_idx %= output_strides[i];
            }
            
            // Map output coordinates to input coordinates using permutation
            std::vector<size_t> input_coords(shape.ndim());
            for (size_t i = 0; i < dims.size(); ++i) {
                input_coords[dims[i]] = output_coords[i];
            }
            
            // Convert input coordinates to linear index
            size_t input_idx = 0;
            for (size_t i = 0; i < shape.ndim(); ++i) {
                input_idx += input_coords[i] * input_strides[i];
            }
            
            result_data[output_idx] = input_data[input_idx];
        }
    } else if (input.dtype() == DataType::kInt32) {
        const int32_t* input_data = input.data_ptr<int32_t>();
        int32_t* result_data = result.data_ptr<int32_t>();
        
        size_t total_elements = permuted_shape.total_size();
        
        for (size_t output_idx = 0; output_idx < total_elements; ++output_idx) {
            std::vector<size_t> output_coords(shape.ndim());
            size_t temp_idx = output_idx;
            
            for (size_t i = 0; i < shape.ndim(); ++i) {
                output_coords[i] = temp_idx / output_strides[i];
                temp_idx %= output_strides[i];
            }
            
            std::vector<size_t> input_coords(shape.ndim());
            for (size_t i = 0; i < dims.size(); ++i) {
                input_coords[dims[i]] = output_coords[i];
            }
            
            size_t input_idx = 0;
            for (size_t i = 0; i < shape.ndim(); ++i) {
                input_idx += input_coords[i] * input_strides[i];
            }
            
            result_data[output_idx] = input_data[input_idx];
        }
    } else {
        throw std::runtime_error("Permute currently only supports Float32 and Int32 data types");
    }
    
    return result;
}

void TensorEngine::validate_binary_op_compatibility(const Tensor& a, const Tensor& b) const {
    // Basic compatibility check
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Tensor data types must match for binary operations");
    }
}

Tensor TensorEngine::convert_dtype(const Tensor& input, DataType target_dtype) {
    if (input.dtype() == target_dtype) {
        return input;  // No conversion needed
    }
    
    // Create output tensor with target data type
    Tensor result(input.shape(), target_dtype);
    size_t num_elements = input.shape().total_size();
    
    // Convert based on source and target types
    if (input.dtype() == DataType::kInt8 && target_dtype == DataType::kFloat32) {
        // INT8 -> Float32
        const int8_t* src_data = input.data_ptr<int8_t>();
        float* dst_data = result.data_ptr<float>();
        
        for (size_t i = 0; i < num_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }
        
    } else if (input.dtype() == DataType::kUInt8 && target_dtype == DataType::kFloat32) {
        // INT4 (stored as UInt8) -> Float32
        const uint8_t* src_data = input.data_ptr<uint8_t>();
        float* dst_data = result.data_ptr<float>();
        
        for (size_t i = 0; i < num_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }
        
    } else if (input.dtype() == DataType::kInt32 && target_dtype == DataType::kFloat32) {
        // INT4 (stored as Int32) -> Float32
        const int32_t* src_data = input.data_ptr<int32_t>();
        float* dst_data = result.data_ptr<float>();
        
        for (size_t i = 0; i < num_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }
        
    } else if (input.dtype() == DataType::kFloat32 && target_dtype == DataType::kInt8) {
        // Float32 -> INT8
        const float* src_data = input.data_ptr<float>();
        int8_t* dst_data = result.data_ptr<int8_t>();
        
        for (size_t i = 0; i < num_elements; ++i) {
            float val = src_data[i];
            val = std::max(-128.0f, std::min(127.0f, val));  // Manual clamp
            dst_data[i] = static_cast<int8_t>(std::round(val));
        }
        
    } else if (input.dtype() == DataType::kFloat32 && target_dtype == DataType::kUInt8) {
        // Float32 -> INT4 (stored as UInt8)
        const float* src_data = input.data_ptr<float>();
        uint8_t* dst_data = result.data_ptr<uint8_t>();
        
        for (size_t i = 0; i < num_elements; ++i) {
            float val = src_data[i];
            val = std::max(0.0f, std::min(15.0f, val));  // Manual clamp
            dst_data[i] = static_cast<uint8_t>(std::round(val));
        }
        
    } else {
        throw std::runtime_error("Unsupported data type conversion: " + 
                               std::to_string(static_cast<int>(input.dtype())) + " -> " +
                               std::to_string(static_cast<int>(target_dtype)));
    }
    
    return result;
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
