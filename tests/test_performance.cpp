/**
 * @file test_performance.cpp
 * @brief Performance tests and benchmarks for TurboInfer.
 * @author J.J.G. Pleunes
 */

#include <gtest/gtest.h>
#include "turboinfer/core/tensor.hpp"
#include "turboinfer/core/tensor_engine.hpp"
#include "turboinfer/util/profiler.hpp"
#include <chrono>
#include <vector>

using namespace turboinfer::core;
using namespace turboinfer::util;

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize profiler
        profiler = std::make_unique<Profiler>();
    }

    void TearDown() override {
        // Cleanup profiler
        profiler.reset();
    }

    std::unique_ptr<Profiler> profiler;
    
    // Helper function to measure execution time
    template<typename Func>
    double measure_time_ms(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0; // Convert to milliseconds
    }
};

TEST_F(PerformanceTest, Tensor_Creation_Performance) {
    const int num_iterations = 1000;
    std::vector<double> times;
    
    // Test tensor creation performance
    for (int i = 0; i < num_iterations; ++i) {
        double time = measure_time_ms([&]() {
            TensorShape shape({100, 100});
            Tensor tensor(shape, DataType::kFloat32);
        });
        times.push_back(time);
    }
    
    // Calculate statistics
    double total_time = 0.0;
    for (double t : times) {
        total_time += t;
    }
    double avg_time = total_time / num_iterations;
    
    // Performance expectations (adjust based on system)
    EXPECT_LT(avg_time, 1.0); // Should be less than 1ms on average
    
    std::cout << "Tensor creation avg time: " << avg_time << " ms" << std::endl;
}

TEST_F(PerformanceTest, Tensor_Copy_Performance) {
    TensorShape shape({500, 500});
    Tensor original(shape, DataType::kFloat32);
    
    const int num_iterations = 100;
    std::vector<double> times;
    
    // Test tensor copying performance
    for (int i = 0; i < num_iterations; ++i) {
        double time = measure_time_ms([&]() {
            Tensor copy(original);
        });
        times.push_back(time);
    }
    
    double total_time = 0.0;
    for (double t : times) {
        total_time += t;
    }
    double avg_time = total_time / num_iterations;
    
    // Large tensor copy should still be reasonable
    EXPECT_LT(avg_time, 50.0); // Should be less than 50ms on average
    
    std::cout << "Tensor copy (500x500) avg time: " << avg_time << " ms" << std::endl;
}

TEST_F(PerformanceTest, Tensor_Slice_Performance) {
    TensorShape shape({1000, 1000});
    Tensor tensor(shape, DataType::kFloat32);
    
    std::vector<std::size_t> start = {100, 100};
    std::vector<std::size_t> end = {900, 900};
    
    const int num_iterations = 1000;
    std::vector<double> times;
    
    // Test slicing performance
    for (int i = 0; i < num_iterations; ++i) {
        double time = measure_time_ms([&]() {
            Tensor sliced = tensor.slice(start, end);
        });
        times.push_back(time);
    }
    
    double total_time = 0.0;
    for (double t : times) {
        total_time += t;
    }
    double avg_time = total_time / num_iterations;
    
    // Slicing should be very fast (mostly metadata operations)
    EXPECT_LT(avg_time, 0.1); // Should be less than 0.1ms on average
    
    std::cout << "Tensor slice avg time: " << avg_time << " ms" << std::endl;
}

TEST_F(PerformanceTest, Memory_Allocation_Scaling) {
    // Test how memory allocation scales with tensor size
    std::vector<size_t> sizes = {100, 200, 500, 1000, 2000};
    
    for (size_t size : sizes) {
        double time = measure_time_ms([&]() {
            TensorShape shape({size, size});
            Tensor tensor(shape, DataType::kFloat32);
        });
        
        size_t total_elements = size * size;
        double elements_per_ms = total_elements / time;
        
        std::cout << "Size " << size << "x" << size 
                  << " (" << total_elements << " elements): " 
                  << time << " ms, " 
                  << elements_per_ms << " elements/ms" << std::endl;
        
        // Should handle reasonable sizes efficiently
        if (size <= 1000) {
            EXPECT_LT(time, 10.0); // Should be less than 10ms for sizes up to 1000x1000
        }
    }
}

TEST_F(PerformanceTest, Profiler_Overhead) {
    const int num_iterations = 10000;
    
    // Measure without profiler
    double time_without = measure_time_ms([&]() {
        for (int i = 0; i < num_iterations; ++i) {
            TensorShape shape({10, 10});
            Tensor tensor(shape, DataType::kFloat32);
        }
    });
    
    // Measure with profiler
    double time_with = measure_time_ms([&]() {
        for (int i = 0; i < num_iterations; ++i) {
            ProfileScope scope(*profiler, "tensor_creation");
            TensorShape shape({10, 10});
            Tensor tensor(shape, DataType::kFloat32);
        }
    });
    
    // Profiler overhead should be minimal
    double overhead_percent = ((time_with - time_without) / time_without) * 100.0;
    
    std::cout << "Profiler overhead: " << overhead_percent << "%" << std::endl;
    
    // Overhead should be reasonable (less than 20%)
    EXPECT_LT(overhead_percent, 20.0);
}

TEST_F(PerformanceTest, Data_Type_Performance_Comparison) {
    TensorShape shape({500, 500});
    const int num_iterations = 50;
    
    std::vector<DataType> types = {
        DataType::kFloat32,
        DataType::kFloat16,
        DataType::kInt32,
        DataType::kInt8
    };
    
    for (auto dtype : types) {
        double time = measure_time_ms([&]() {
            for (int i = 0; i < num_iterations; ++i) {
                Tensor tensor(shape, dtype);
            }
        });
        
        double avg_time = time / num_iterations;
        size_t bytes_per_element = data_type_size(dtype);
        size_t total_bytes = shape.total_size() * bytes_per_element;
        
        std::cout << "DataType " << data_type_to_string(dtype) 
                  << " (" << bytes_per_element << " bytes/element): "
                  << avg_time << " ms avg, "
                  << total_bytes << " total bytes" << std::endl;
    }
}

TEST_F(PerformanceTest, Matrix_Multiply_Performance_Placeholder) {
    // Note: This tests the placeholder implementation
    // Real performance tests will be added when actual GEMM is implemented
    
    TensorEngine engine;
    TensorShape shape_a({100, 200});
    TensorShape shape_b({200, 150});
    TensorShape shape_c({100, 150});
    
    Tensor a(shape_a, DataType::kFloat32);
    Tensor b(shape_b, DataType::kFloat32);
    Tensor c(shape_c, DataType::kFloat32);
    
    double time = measure_time_ms([&]() {
        engine.matrix_multiply(a, b, c);
    });
    
    // Since this is a placeholder, just ensure it doesn't crash
    EXPECT_GT(time, 0.0);
    std::cout << "Matrix multiply (placeholder) time: " << time << " ms" << std::endl;
}

TEST_F(PerformanceTest, Large_Tensor_Stress_Test) {
    // Stress test with larger tensors
    const size_t large_size = 2048;
    
    double creation_time = measure_time_ms([&]() {
        TensorShape shape({large_size, large_size});
        Tensor tensor(shape, DataType::kFloat32);
        
        // Test some operations on large tensor
        std::vector<std::size_t> start = {0, 0};
        std::vector<std::size_t> end = {100, 100};
        Tensor slice = tensor.slice(start, end);
    });
    
    size_t total_elements = large_size * large_size;
    size_t total_bytes = total_elements * 4; // float32
    
    std::cout << "Large tensor (" << large_size << "x" << large_size << "): "
              << creation_time << " ms, "
              << (total_bytes / 1024 / 1024) << " MB" << std::endl;
    
    // Should handle large tensors reasonably (adjust based on available memory)
    EXPECT_LT(creation_time, 1000.0); // Should be less than 1 second
}
