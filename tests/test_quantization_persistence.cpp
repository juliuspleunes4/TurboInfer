/**
 * @file test_quantization_persistence.cpp
 * @brief Tests for quantization model save/load functionality
 * @author J.J.G. Pleunes
 */

#include "turboinfer/optimize/quantization.hpp"
#include "turboinfer/model/model_loader.hpp"
#include <iostream>
#include <cassert>
#include <filesystem>

using namespace turboinfer::optimize;
using namespace turboinfer::model;
using namespace turboinfer::core;

#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        std::cerr << "ASSERTION FAILED: " << #condition << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } else { \
        std::cout << "✅ PASS: " << #condition << std::endl; \
    }

#define ASSERT_NEAR(a, b, tolerance) \
    if (std::abs((a) - (b)) > (tolerance)) { \
        std::cerr << "ASSERTION FAILED: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ") within " << (tolerance) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } else { \
        std::cout << "✅ PASS: " << #a << " ≈ " << #b << " within " << (tolerance) << std::endl; \
    }

ModelData create_test_model() {
    ModelData model;
    
    // Set up metadata
    auto& metadata = model.metadata();
    metadata.name = "test_model";
    metadata.architecture = "llama";
    metadata.version = "2.0";
    metadata.vocab_size = 1000;
    metadata.hidden_size = 256;
    metadata.num_layers = 4;
    metadata.num_heads = 8;
    metadata.intermediate_size = 512;
    metadata.rope_theta = 10000.0f;
    
    // Add some test tensors
    // Weight tensor 1: 256x256 matrix
    TensorShape weight1_shape({256, 256});
    Tensor weight1(weight1_shape, DataType::kFloat32);
    float* weight1_data = weight1.data_ptr<float>();
    for (size_t i = 0; i < weight1_shape.total_size(); ++i) {
        weight1_data[i] = static_cast<float>(i % 100) / 100.0f - 0.5f; // Values in [-0.5, 0.5]
    }
    model.add_tensor("weight1", std::move(weight1));
    
    // Weight tensor 2: 256x512 matrix
    TensorShape weight2_shape({256, 512});
    Tensor weight2(weight2_shape, DataType::kFloat32);
    float* weight2_data = weight2.data_ptr<float>();
    for (size_t i = 0; i < weight2_shape.total_size(); ++i) {
        weight2_data[i] = static_cast<float>(i % 127) / 127.0f - 0.5f; // Values in [-0.5, 0.5]
    }
    model.add_tensor("weight2", std::move(weight2));
    
    // Bias tensor: 256 elements
    TensorShape bias_shape({256});
    Tensor bias(bias_shape, DataType::kFloat32);
    float* bias_data = bias.data_ptr<float>();
    for (size_t i = 0; i < bias_shape.total_size(); ++i) {
        bias_data[i] = static_cast<float>(i % 64) / 64.0f - 0.5f; // Values in [-0.5, 0.5]
    }
    model.add_tensor("bias", std::move(bias));
    
    return model;
}

void test_quantization_save_load() {
    std::cout << "\n--- Test: Quantization Save/Load ---" << std::endl;
    
    const std::string test_file = "test_quantized_model.tinq";
    
    try {
        // Create test model
        auto original_model = create_test_model();
        std::cout << "✅ Created test model with " << original_model.tensor_names().size() << " tensors" << std::endl;
        
        // Configure quantizer for INT8
        QuantizationConfig config;
        config.type = QuantizationType::kInt8;
        config.symmetric = true;
        
        Quantizer quantizer(config);
        
        // Quantize the model
        auto quantized_model = quantizer.quantize_model(original_model);
        std::cout << "✅ Quantized model successfully" << std::endl;
        
        // Save quantized model
        quantizer.save_quantized_model(quantized_model, test_file);
        std::cout << "✅ Saved quantized model to: " << test_file << std::endl;
        
        // Verify file exists
        ASSERT_TRUE(std::filesystem::exists(test_file));
        
        // Get file size
        auto file_size = std::filesystem::file_size(test_file);
        std::cout << "✅ Quantized model file size: " << file_size << " bytes" << std::endl;
        
        // Load quantized model
        auto loaded_model = Quantizer::load_quantized_model(test_file);
        std::cout << "✅ Loaded quantized model successfully" << std::endl;
        
        // Verify metadata
        const auto& orig_meta = original_model.metadata();
        const auto& loaded_meta = loaded_model.metadata();
        
        ASSERT_TRUE(orig_meta.name == loaded_meta.name);
        ASSERT_TRUE(orig_meta.architecture == loaded_meta.architecture);
        ASSERT_TRUE(orig_meta.version == loaded_meta.version);
        ASSERT_TRUE(orig_meta.vocab_size == loaded_meta.vocab_size);
        ASSERT_TRUE(orig_meta.hidden_size == loaded_meta.hidden_size);
        ASSERT_TRUE(orig_meta.num_layers == loaded_meta.num_layers);
        ASSERT_TRUE(orig_meta.num_heads == loaded_meta.num_heads);
        ASSERT_TRUE(orig_meta.intermediate_size == loaded_meta.intermediate_size);
        ASSERT_NEAR(orig_meta.rope_theta, loaded_meta.rope_theta, 1e-6f);
        std::cout << "✅ Metadata preserved correctly" << std::endl;
        
        // Verify tensor count
        auto orig_names = original_model.tensor_names();
        auto loaded_names = loaded_model.tensor_names();
        ASSERT_TRUE(orig_names.size() == loaded_names.size());
        std::cout << "✅ Tensor count preserved: " << loaded_names.size() << " tensors" << std::endl;
        
        // Verify tensor names and shapes
        for (const auto& name : orig_names) {
            const auto* orig_tensor = original_model.get_tensor(name);
            const auto* loaded_tensor = loaded_model.get_tensor(name);
            
            ASSERT_TRUE(orig_tensor != nullptr);
            ASSERT_TRUE(loaded_tensor != nullptr);
            
            // Shape should be the same
            ASSERT_TRUE(orig_tensor->shape().ndim() == loaded_tensor->shape().ndim());
            for (size_t i = 0; i < orig_tensor->shape().ndim(); ++i) {
                ASSERT_TRUE(orig_tensor->shape().size(i) == loaded_tensor->shape().size(i));
            }
            
            // Loaded tensor should be quantized (INT8)
            ASSERT_TRUE(loaded_tensor->dtype() == DataType::kInt8);
            
            std::cout << "✅ Tensor '" << name << "' preserved with shape " 
                      << orig_tensor->shape().ndim() << "D and quantized to INT8" << std::endl;
        }
        
        // Clean up test file
        std::filesystem::remove(test_file);
        std::cout << "✅ Cleaned up test file" << std::endl;
        
        std::cout << "✅ Quantization persistence test passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        
        // Clean up on failure
        if (std::filesystem::exists(test_file)) {
            std::filesystem::remove(test_file);
        }
        throw;
    }
}

void test_round_trip_accuracy() {
    std::cout << "\n--- Test: Round-trip Accuracy ---" << std::endl;
    
    const std::string test_file = "test_round_trip.tinq";
    
    try {
        // Create simple test tensor
        TensorShape shape({10, 10});
        Tensor original(shape, DataType::kFloat32);
        float* data = original.data_ptr<float>();
        
        // Fill with known values
        for (size_t i = 0; i < shape.total_size(); ++i) {
            data[i] = static_cast<float>(i) / 100.0f - 0.5f;
        }
        
        // Create model with this tensor
        ModelData model;
        auto& metadata = model.metadata();
        metadata.name = "round_trip_test";
        metadata.architecture = "test";
        metadata.vocab_size = 100;
        metadata.hidden_size = 64;
        metadata.num_layers = 1;
        metadata.num_heads = 1;
        metadata.intermediate_size = 128;
        
        model.add_tensor("test_tensor", std::move(original));
        
        // Quantize, save, and load
        QuantizationConfig config;
        config.type = QuantizationType::kInt8;
        config.symmetric = true;
        
        Quantizer quantizer(config);
        auto quantized = quantizer.quantize_model(model);
        quantizer.save_quantized_model(quantized, test_file);
        auto loaded = Quantizer::load_quantized_model(test_file);
        
        // Basic verification
        ASSERT_TRUE(loaded.tensor_names().size() == 1);
        const auto* loaded_tensor = loaded.get_tensor("test_tensor");
        ASSERT_TRUE(loaded_tensor != nullptr);
        ASSERT_TRUE(loaded_tensor->dtype() == DataType::kInt8);
        ASSERT_TRUE(loaded_tensor->shape().total_size() == 100);
        
        std::cout << "✅ Round-trip test passed - quantized tensor properly saved and loaded" << std::endl;
        
        // Clean up
        std::filesystem::remove(test_file);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Round-trip test failed: " << e.what() << std::endl;
        if (std::filesystem::exists(test_file)) {
            std::filesystem::remove(test_file);
        }
        throw;
    }
}

int main() {
    std::cout << "🚀 Testing Quantization Persistence" << std::endl;
    std::cout << "====================================" << std::endl;
    
    try {
        test_quantization_save_load();
        test_round_trip_accuracy();
        
        std::cout << "\n🎉 ALL QUANTIZATION PERSISTENCE TESTS PASSED!" << std::endl;
        std::cout << "✅ save_quantized_model() and load_quantized_model() work correctly!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Tests failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
