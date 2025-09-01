#include "turboinfer/model/model_loader.hpp"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdexcept>

int main() {
    try {
        std::cout << "Testing PyTorch model loader..." << std::endl;
        
        // Test with a non-existent PyTorch file to see if the ZIP parsing works
        std::string test_file = "fake_model.pth";
        
        // Create a minimal fake PyTorch ZIP file for testing
        std::ofstream fake_file(test_file, std::ios::binary);
        if (fake_file.is_open()) {
            // Write a minimal ZIP file header
            uint32_t local_file_header = 0x04034b50;  // ZIP local file header signature
            fake_file.write(reinterpret_cast<const char*>(&local_file_header), sizeof(local_file_header));
            fake_file.close();
        }
        
        // Test format detection
        turboinfer::model::ModelFormat format = turboinfer::model::ModelLoader::detect_format(test_file);
        std::cout << "Detected format: " << static_cast<int>(format) << std::endl;
        
        // Test loading (should use the mock implementation)
        try {
            auto model_data = turboinfer::model::ModelLoader::load(test_file, turboinfer::model::ModelFormat::kPyTorch);
            std::cout << "✅ Successfully loaded PyTorch model!" << std::endl;
            std::cout << "Model name: " << model_data.metadata().name << std::endl;
            std::cout << "Architecture: " << model_data.metadata().architecture << std::endl;
            std::cout << "Hidden size: " << model_data.metadata().hidden_size << std::endl;
            std::cout << "Number of layers: " << model_data.metadata().num_layers << std::endl;
            std::cout << "Number of tensors: " << model_data.tensor_names().size() << std::endl;
            
            // Cleanup
            std::remove(test_file.c_str());
            
        } catch (const std::exception& e) {
            std::cout << "❌ Failed to load PyTorch model: " << e.what() << std::endl;
            std::remove(test_file.c_str());
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "✅ PyTorch loader test completed successfully!" << std::endl;
    return 0;
}
