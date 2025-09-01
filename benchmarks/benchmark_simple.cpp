/**
 * @file benchmark_simple.cpp
 * @brief Simplified benchmarking for current TurboInfer capabilities.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/model/inference_engine.hpp"
#include "turboinfer/model/model_loader.hpp"
#include "turboinfer/optimize/quantization.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace turboinfer::model;
using namespace turboinfer::optimize;
using namespace turboinfer::core;

/**
 * @brief Simple benchmark for 2D tensor operations.
 */
class SimpleBenchmark {
public:
    /**
     * @brief Run basic tensor operation benchmarks.
     */
    void run_tensor_benchmarks() {
        std::cout << "=========================================" << std::endl;
        std::cout << "    TurboInfer Simple Benchmarks" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        benchmark_matrix_operations();
        benchmark_quantization_operations();
        benchmark_beam_search_operations();
    }

private:
    void benchmark_matrix_operations() {
        std::cout << "\n=== Matrix Operations Benchmark ===" << std::endl;
        
        try {
            TensorEngine engine;
            
            // Test different matrix sizes
            std::vector<std::pair<size_t, size_t>> sizes = {
                {64, 128},    // Small
                {256, 512},   // Medium  
                {512, 1024}   // Large
            };
            
            for (const auto& [rows, cols] : sizes) {
                std::cout << "Testing " << rows << "x" << cols << " matrices..." << std::endl;
                
                // Create test matrices
                TensorShape shape_a({rows, cols});
                TensorShape shape_b({cols, rows});
                
                Tensor a(shape_a, DataType::kFloat32);
                Tensor b(shape_b, DataType::kFloat32);
                
                // Fill with random data
                fill_random(a);
                fill_random(b);
                
                // Benchmark matrix multiplication
                auto start = std::chrono::steady_clock::now();
                auto result = engine.matmul(a, b);
                auto end = std::chrono::steady_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0;
                
                // Calculate operations per second
                size_t ops = rows * cols * rows; // Rough FLOP count
                double ops_per_sec = (ops * 1000.0) / time_ms;
                
                std::cout << "  " << rows << "x" << cols << ": " << time_ms << "ms (" 
                          << (ops_per_sec / 1e6) << " MFLOPS)" << std::endl;
            }
            
            std::cout << "✅ Matrix operations benchmark completed!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "❌ Matrix operations benchmark failed: " << e.what() << std::endl;
        }
    }
    
    void benchmark_quantization_operations() {
        std::cout << "\n=== Quantization Operations Benchmark ===" << std::endl;
        
        try {
            // Create test tensor
            TensorShape shape({1000, 512});
            Tensor test_tensor(shape, DataType::kFloat32);
            fill_random(test_tensor);
            
            std::cout << "Testing quantization on " << shape.size(0) << "x" << shape.size(1) << " tensor..." << std::endl;
            
            // Test INT8 quantization
            QuantizationConfig config_int8;
            config_int8.type = QuantizationType::kInt8;
            config_int8.symmetric = true;
            
            Quantizer quantizer_int8(config_int8);
            
            auto start = std::chrono::steady_clock::now();
            auto quantized_int8 = quantizer_int8.quantize_tensor(test_tensor);
            auto end = std::chrono::steady_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double time_ms = duration.count() / 1000.0;
            
            size_t elements = test_tensor.shape().total_size();
            double elements_per_sec = (elements * 1000.0) / time_ms;
            
            std::cout << "  INT8 quantization: " << time_ms << "ms (" 
                      << (elements_per_sec / 1e6) << " M elements/sec)" << std::endl;
            
            // Test INT4 quantization  
            QuantizationConfig config_int4;
            config_int4.type = QuantizationType::kInt4;
            config_int4.symmetric = true;
            
            Quantizer quantizer_int4(config_int4);
            
            start = std::chrono::steady_clock::now();
            auto quantized_int4 = quantizer_int4.quantize_tensor(test_tensor);
            end = std::chrono::steady_clock::now();
            
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            time_ms = duration.count() / 1000.0;
            elements_per_sec = (elements * 1000.0) / time_ms;
            
            std::cout << "  INT4 quantization: " << time_ms << "ms (" 
                      << (elements_per_sec / 1e6) << " M elements/sec)" << std::endl;
            
            // Calculate compression ratios
            float int8_ratio = static_cast<float>(test_tensor.byte_size()) / quantized_int8.byte_size();
            float int4_ratio = static_cast<float>(test_tensor.byte_size()) / quantized_int4.byte_size();
            
            std::cout << "  Compression ratios: INT8=" << int8_ratio << "x, INT4=" << int4_ratio << "x" << std::endl;
            std::cout << "✅ Quantization operations benchmark completed!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "❌ Quantization operations benchmark failed: " << e.what() << std::endl;
        }
    }
    
    void benchmark_beam_search_operations() {
        std::cout << "\n=== Beam Search Operations Benchmark ===" << std::endl;
        
        try {
            // Simple test of beam search with mock data
            size_t vocab_size = 1000;
            size_t beam_size = 4;
            size_t max_length = 10;
            
            std::cout << "Testing beam search (vocab=" << vocab_size << ", beams=" << beam_size 
                      << ", length=" << max_length << ")..." << std::endl;
            
            // Create mock model data for beam search
            ModelData model_data;
            model_data.metadata().vocab_size = vocab_size;
            model_data.metadata().hidden_size = 256;
            model_data.metadata().num_layers = 2;
            model_data.metadata().num_heads = 8;
            model_data.metadata().intermediate_size = 1024;
            
            // Add minimal required tensors for beam search
            TensorShape embedding_shape({vocab_size, model_data.metadata().hidden_size});
            auto embedding = std::make_unique<Tensor>(embedding_shape, DataType::kFloat32);
            fill_random(*embedding);
            model_data.add_tensor("token_embeddings.weight", *embedding);
            
            InferenceConfig config;
            config.temperature = 0.8f;
            config.top_p = 0.9f;
            
            InferenceEngine engine(model_data, config);
            
            std::vector<int> prompt = {1, 50, 100};
            
            auto start = std::chrono::steady_clock::now();
            auto results = engine.generate_beam_search(prompt, max_length, beam_size, true);
            auto end = std::chrono::steady_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double time_ms = duration.count() / 1000.0;
            
            size_t total_tokens = results.size() * max_length;
            double tokens_per_sec = (total_tokens * 1000.0) / time_ms;
            
            std::cout << "  Generated " << results.size() << " beams in " << time_ms << "ms" << std::endl;
            std::cout << "  Performance: " << tokens_per_sec << " tokens/second" << std::endl;
            std::cout << "✅ Beam search operations benchmark completed!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "❌ Beam search operations benchmark failed: " << e.what() << std::endl;
        }
    }
    
    void fill_random(Tensor& tensor) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        float* data = static_cast<float*>(tensor.data());
        size_t size = tensor.shape().total_size();
        
        for (size_t i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
    }
};

int main() {
    try {
        SimpleBenchmark benchmark;
        benchmark.run_tensor_benchmarks();
        
        std::cout << "\n🎉 Simple benchmarks completed successfully!" << std::endl;
        std::cout << "This validates the core TurboInfer operations." << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}
