#include "turboinfer/turboinfer.hpp"
#include <chrono>
#include <iostream>
#include <vector>

using namespace turboinfer;

int main() {
    if (!turboinfer::initialize(true)) {
        std::cerr << "Failed to initialize TurboInfer" << std::endl;
        return 1;
    }
    
    std::cout << "=== TurboInfer SIMD Performance Test ===" << std::endl;
    std::cout << "Build Info: " << turboinfer::build_info() << std::endl;
    
    // Test matrix multiplication performance with larger matrices
    const size_t M = 128, K = 128, N = 128;
    
    core::TensorShape shape_a({M, K});
    core::TensorShape shape_b({K, N});
    
    core::Tensor a(shape_a, core::DataType::kFloat32);
    core::Tensor b(shape_b, core::DataType::kFloat32);
    
    // Fill with test data
    float* a_data = a.data_ptr<float>();
    float* b_data = b.data_ptr<float>();
    
    for (size_t i = 0; i < M * K; ++i) {
        a_data[i] = static_cast<float>(i % 100) / 100.0f;
    }
    
    for (size_t i = 0; i < K * N; ++i) {
        b_data[i] = static_cast<float>(i % 100) / 100.0f;
    }
    
    core::TensorEngine engine;
    
    // Warm up
    auto warmup = engine.matmul(a, b);
    
    // Performance test - multiple runs
    const int num_runs = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; ++i) {
        auto result = engine.matmul(a, b);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Matrix multiplication (" << M << "x" << K << ") * (" << K << "x" << N << ")" << std::endl;
    std::cout << "Average time per operation: " << duration.count() / num_runs << " microseconds" << std::endl;
    
    // Test element-wise operations
    core::TensorShape vec_shape({100000}); // Large vector for SIMD benefit
    core::Tensor vec_a(vec_shape, core::DataType::kFloat32);
    core::Tensor vec_b(vec_shape, core::DataType::kFloat32);
    
    float* va_data = vec_a.data_ptr<float>();
    float* vb_data = vec_b.data_ptr<float>();
    
    for (size_t i = 0; i < 100000; ++i) {
        va_data[i] = static_cast<float>(i % 1000) / 1000.0f;
        vb_data[i] = static_cast<float>((i + 500) % 1000) / 1000.0f;
    }
    
    // Test vectorized addition
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; ++i) {
        auto result = engine.add(vec_a, vec_b);
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Element-wise addition (100,000 elements)" << std::endl;
    std::cout << "Average time per operation: " << duration.count() / num_runs << " microseconds" << std::endl;
    
    // Test ReLU activation
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; ++i) {
        auto result = engine.relu(vec_a);
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "ReLU activation (100,000 elements)" << std::endl;
    std::cout << "Average time per operation: " << duration.count() / num_runs << " microseconds" << std::endl;
    
    std::cout << "=== SIMD Performance Test Completed ===" << std::endl;
    
    turboinfer::shutdown();
    return 0;
}
