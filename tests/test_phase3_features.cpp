#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Testing Phase 3 Enhanced Features ===" << std::endl;
    
    // Initialize TurboInfer
    if (!turboinfer::initialize(true)) {
        std::cerr << "Failed to initialize TurboInfer" << std::endl;
        return 1;
    }
    
    try {
        // Create a sample model for testing
        turboinfer::model::ModelData test_model;
        
        // Set up metadata
        auto& metadata = test_model.metadata();
        metadata.name = "test_model_phase3";
        metadata.architecture = "llama";
        metadata.version = "1.0";
        metadata.vocab_size = 32000;
        metadata.hidden_size = 4096;
        metadata.num_layers = 32;
        metadata.num_heads = 32;
        metadata.intermediate_size = 11008;
        metadata.rope_theta = 10000.0f;
        
        // Add some sample tensors
        std::vector<size_t> embedding_dims = {32000, 4096};
        turboinfer::core::TensorShape embedding_shape(embedding_dims);
        turboinfer::core::Tensor embedding_tensor(embedding_shape, turboinfer::core::DataType::kFloat32);
        test_model.add_tensor("token_embeddings.weight", std::move(embedding_tensor));
        
        std::vector<size_t> norm_dims = {4096};
        turboinfer::core::TensorShape norm_shape(norm_dims);
        turboinfer::core::Tensor norm_tensor(norm_shape, turboinfer::core::DataType::kFloat32);
        test_model.add_tensor("norm.weight", std::move(norm_tensor));
        
        // Test new configuration handling
        test_model.set_config_param("temperature", "0.7");
        test_model.set_config_param("top_p", "0.9");
        test_model.set_config_param("max_seq_len", "2048");
        
        // Test new enhanced features
        std::cout << "\n=== Model Summary ===" << std::endl;
        std::cout << test_model.get_model_summary() << std::endl;
        
        std::cout << "=== Configuration Parameters ===" << std::endl;
        std::cout << "Temperature: " << test_model.get_config_param("temperature") << std::endl;
        std::cout << "Top-P: " << test_model.get_config_param("top_p") << std::endl;
        std::cout << "Max Seq Length: " << test_model.get_config_param("max_seq_len") << std::endl;
        std::cout << "Unknown param: " << test_model.get_config_param("unknown", "default_value") << std::endl;
        
        std::cout << "\n=== Model Validation ===" << std::endl;
        bool is_valid = test_model.validate();
        std::cout << "Model is valid: " << (is_valid ? "Yes" : "No") << std::endl;
        
        std::cout << "\n=== Memory Usage ===" << std::endl;
        std::cout << "Total memory: " << test_model.get_memory_usage_string() << std::endl;
        std::cout << "Raw bytes: " << test_model.total_memory_usage() << " bytes" << std::endl;
        
        // Test model format error handling
        std::cout << "\n=== Format Error Handling Tests ===" << std::endl;
        
        try {
            turboinfer::model::ModelLoader::load("fake_model.pt");
        } catch (const std::exception& e) {
            std::cout << "PyTorch error: " << e.what() << std::endl;
        }
        
        try {
            turboinfer::model::ModelLoader::load("fake_model.onnx");
        } catch (const std::exception& e) {
            std::cout << "ONNX error: " << e.what() << std::endl;
        }
        
        std::cout << "\n✅ Phase 3 Enhanced Features Test Completed Successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        turboinfer::shutdown();
        return 1;
    }
    
    turboinfer::shutdown();
    return 0;
}
