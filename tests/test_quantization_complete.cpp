/**
 * @file test_quantization_complete.cpp
 * @brief Comprehensive test for Phase 5 quantization functionality.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/optimize/quantization.hpp"
#include "turboinfer/core/tensor.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cmath>

using namespace turboinfer::optimize;
using namespace turboinfer::core;

void test_int8_quantization() {
    std::cout << "\n=== Testing INT8 Quantization ===" << std::endl;
    
    // Create test tensor with known values
    TensorShape shape({4, 4});
    Tensor input_tensor(shape, DataType::kFloat32);
    
    // Fill with test data: values from -10.0 to 10.0
    float* data = static_cast<float*>(input_tensor.data());
    for (size_t i = 0; i < shape.total_size(); ++i) {
        data[i] = -10.0f + (20.0f * i / (shape.total_size() - 1));
    }
    
    std::cout << "Input tensor range: " << data[0] << " to " << data[shape.total_size()-1] << std::endl;
    
    // Configure INT8 quantization
    QuantizationConfig config;
    config.type = QuantizationType::kInt8;
    config.symmetric = true;
    
    Quantizer quantizer(config);
    
    // Test quantization
    auto start = std::chrono::steady_clock::now();
    auto quantized = quantizer.quantize_tensor(input_tensor);
    auto quant_info = quantizer.calculate_quantization_info(input_tensor);
    auto end = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Quantization completed in " << duration.count() << " μs" << std::endl;
    std::cout << "Original size: " << quant_info.original_size_bytes << " bytes" << std::endl;
    std::cout << "Quantized size: " << quant_info.quantized_size_bytes << " bytes" << std::endl;
    std::cout << "Compression ratio: " << quant_info.compression_ratio << "x" << std::endl;
    std::cout << "Scale: " << quant_info.scales[0] << std::endl;
    
    // Test dequantization
    auto dequantized = quantizer.dequantize_tensor(quantized, quant_info);
    
    // Calculate error
    float* orig_data = static_cast<float*>(input_tensor.data());
    float* dequant_data = static_cast<float*>(dequantized.data());
    
    float max_error = 0.0f;
    float avg_error = 0.0f;
    for (size_t i = 0; i < shape.total_size(); ++i) {
        float error = std::abs(orig_data[i] - dequant_data[i]);
        max_error = std::max(max_error, error);
        avg_error += error;
    }
    avg_error /= shape.total_size();
    
    std::cout << "Max reconstruction error: " << max_error << std::endl;
    std::cout << "Avg reconstruction error: " << avg_error << std::endl;
    
    // Verify compression ratio
    assert(quant_info.compression_ratio >= 3.8f); // Should be ~4x for float32->int8
    assert(max_error < 1.0f); // Error should be reasonable
    
    std::cout << "✅ INT8 quantization test passed!" << std::endl;
}

void test_int4_quantization() {
    std::cout << "\n=== Testing INT4 Quantization ===" << std::endl;
    
    // Create test tensor with smaller range for INT4
    TensorShape shape({3, 3});
    Tensor input_tensor(shape, DataType::kFloat32);
    
    // Fill with test data: values from -2.0 to 2.0
    float* data = static_cast<float*>(input_tensor.data());
    for (size_t i = 0; i < shape.total_size(); ++i) {
        data[i] = -2.0f + (4.0f * i / (shape.total_size() - 1));
    }
    
    std::cout << "Input tensor range: " << data[0] << " to " << data[shape.total_size()-1] << std::endl;
    
    // Configure INT4 quantization
    QuantizationConfig config;
    config.type = QuantizationType::kInt4;
    config.symmetric = true;
    
    Quantizer quantizer(config);
    
    // Test quantization
    auto quantized = quantizer.quantize_tensor(input_tensor);
    auto quant_info = quantizer.calculate_quantization_info(input_tensor);
    
    std::cout << "Original size: " << quant_info.original_size_bytes << " bytes" << std::endl;
    std::cout << "Quantized size: " << quant_info.quantized_size_bytes << " bytes" << std::endl;
    std::cout << "Compression ratio: " << quant_info.compression_ratio << "x" << std::endl;
    std::cout << "Scale: " << quant_info.scales[0] << std::endl;
    
    // Test dequantization
    auto dequantized = quantizer.dequantize_tensor(quantized, quant_info);
    
    // Calculate error
    float* orig_data = static_cast<float*>(input_tensor.data());
    float* dequant_data = static_cast<float*>(dequantized.data());
    
    float max_error = 0.0f;
    for (size_t i = 0; i < shape.total_size(); ++i) {
        float error = std::abs(orig_data[i] - dequant_data[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max reconstruction error: " << max_error << std::endl;
    
    // Verify compression ratio
    assert(quant_info.compression_ratio >= 7.0f); // Should be ~8x for float32->int4
    assert(max_error < 1.0f); // Error should be reasonable
    
    std::cout << "✅ INT4 quantization test passed!" << std::endl;
}

void test_asymmetric_quantization() {
    std::cout << "\n=== Testing Asymmetric Quantization ===" << std::endl;
    
    // Create test tensor with asymmetric range (values shifted from center)
    TensorShape shape({2, 3});
    Tensor input_tensor(shape, DataType::kFloat32);
    
    // Fill with values: 1.0 to 6.0 (asymmetric around zero)
    float* data = static_cast<float*>(input_tensor.data());
    for (size_t i = 0; i < shape.total_size(); ++i) {
        data[i] = 1.0f + (5.0f * i / (shape.total_size() - 1));
    }
    
    std::cout << "Input tensor range: " << data[0] << " to " << data[shape.total_size()-1] << std::endl;
    
    // Configure asymmetric INT8 quantization
    QuantizationConfig config;
    config.type = QuantizationType::kInt8;
    config.symmetric = false; // Asymmetric
    
    Quantizer quantizer(config);
    
    // Test quantization
    auto quantized = quantizer.quantize_tensor(input_tensor);
    auto quant_info = quantizer.calculate_quantization_info(input_tensor);
    
    std::cout << "Scale: " << quant_info.scales[0] << std::endl;
    std::cout << "Zero point: " << quant_info.zero_points[0] << std::endl;
    
    // Test dequantization
    auto dequantized = quantizer.dequantize_tensor(quantized, quant_info);
    
    // Calculate error
    float* orig_data = static_cast<float*>(input_tensor.data());
    float* dequant_data = static_cast<float*>(dequantized.data());
    
    float max_error = 0.0f;
    for (size_t i = 0; i < shape.total_size(); ++i) {
        float error = std::abs(orig_data[i] - dequant_data[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max reconstruction error: " << max_error << std::endl;
    
    // Debug: print some values to see what's happening
    std::cout << "Original values: ";
    for (size_t i = 0; i < std::min(size_t(3), shape.total_size()); ++i) {
        std::cout << orig_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Reconstructed values: ";
    for (size_t i = 0; i < std::min(size_t(3), shape.total_size()); ++i) {
        std::cout << dequant_data[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify that zero point is used (non-zero for asymmetric with offset values)
    assert(std::abs(quant_info.zero_points[0]) > 0.1f); // Should be significant for 1-6 range
    assert(max_error < 3.0f); // Allow reasonable quantization error for int8
    
    std::cout << "✅ Asymmetric quantization test passed!" << std::endl;
}

void test_quantization_performance() {
    std::cout << "\n=== Testing Quantization Performance ===" << std::endl;
    
    // Create larger tensor for performance testing
    TensorShape shape({128, 256}); // ~128K elements
    Tensor input_tensor(shape, DataType::kFloat32);
    
    // Fill with random-like data
    float* data = static_cast<float*>(input_tensor.data());
    for (size_t i = 0; i < shape.total_size(); ++i) {
        data[i] = -1.0f + (2.0f * (i % 1000) / 1000.0f); // -1 to 1
    }
    
    std::cout << "Testing with " << shape.total_size() << " elements" << std::endl;
    
    QuantizationConfig config;
    config.type = QuantizationType::kInt8;
    config.symmetric = true;
    
    Quantizer quantizer(config);
    
    // Measure quantization performance
    auto start = std::chrono::steady_clock::now();
    auto quantized = quantizer.quantize_tensor(input_tensor);
    auto quant_time = std::chrono::steady_clock::now();
    auto quant_info = quantizer.calculate_quantization_info(input_tensor);
    auto dequantized = quantizer.dequantize_tensor(quantized, quant_info);
    auto end = std::chrono::steady_clock::now();
    
    auto quant_duration = std::chrono::duration_cast<std::chrono::microseconds>(quant_time - start);
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Quantization time: " << quant_duration.count() << " μs" << std::endl;
    std::cout << "Total time (quant + dequant): " << total_duration.count() << " μs" << std::endl;
    
    float throughput = (shape.total_size() * 1000000.0f) / total_duration.count(); // elements/second
    std::cout << "Throughput: " << throughput << " elements/second" << std::endl;
    
    // Verify memory savings
    size_t memory_saved = quant_info.original_size_bytes - quant_info.quantized_size_bytes;
    float memory_savings_percent = (100.0f * memory_saved) / quant_info.original_size_bytes;
    
    std::cout << "Memory saved: " << memory_saved << " bytes (" << memory_savings_percent << "%)" << std::endl;
    
    assert(throughput > 100000.0f); // Should process >100K elements/second
    assert(memory_savings_percent > 70.0f); // Should save >70% memory
    
    std::cout << "✅ Performance test passed!" << std::endl;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "      TurboInfer Phase 5: Quantization Tests" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    try {
        test_int8_quantization();
        test_int4_quantization();
        test_asymmetric_quantization();
        test_quantization_performance();
        
        std::cout << "\n🎉 All quantization tests passed!" << std::endl;
        std::cout << "\nPhase 5 Progress:" << std::endl;
        std::cout << "✅ INT8 quantization with symmetric/asymmetric modes" << std::endl;
        std::cout << "✅ INT4 quantization with compression" << std::endl;
        std::cout << "✅ Quantization/dequantization pipeline" << std::endl;
        std::cout << "✅ Performance optimization and memory savings" << std::endl;
        std::cout << "✅ Comprehensive error analysis and validation" << std::endl;
        
        std::cout << "\n🚀 Phase 5 Feature 1: Quantization Algorithms - COMPLETE!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Quantization tests failed: " << e.what() << std::endl;
        return 1;
    }
}
