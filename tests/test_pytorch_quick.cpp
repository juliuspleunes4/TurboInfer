/**
 * @brief Quick test to verify PyTorch loader functionality
 */

#include "turboinfer/model/model_loader.hpp"
#include <iostream>
#include <fstream>
#include <cstdio>

int main() {
    std::cout << "🔍 Quick PyTorch Loader Test\n" << std::endl;
    
    try {
        // Test 1: Format detection
        std::cout << "1. Testing PyTorch format detection..." << std::endl;
        
        // Create a minimal PyTorch file
        std::ofstream test_file("quick_test.pth", std::ios::binary);
        uint32_t zip_header = 0x04034b50;  // ZIP local file header signature
        test_file.write(reinterpret_cast<const char*>(&zip_header), sizeof(zip_header));
        
        // Write some fake data
        std::vector<char> fake_data(1024, 'A');
        test_file.write(fake_data.data(), fake_data.size());
        test_file.close();
        
        auto format = turboinfer::model::ModelLoader::detect_format("quick_test.pth");
        if (format == turboinfer::model::ModelFormat::kPyTorch) {
            std::cout << "   ✅ PyTorch format detected correctly" << std::endl;
        } else {
            std::cout << "   ❌ Format detection failed" << std::endl;
            return 1;
        }
        
        // Test 2: Model loading
        std::cout << "\n2. Testing PyTorch model loading..." << std::endl;
        
        auto model_data = turboinfer::model::ModelLoader::load("quick_test.pth");
        const auto& metadata = model_data.metadata();
        
        std::cout << "   Architecture: " << metadata.architecture << std::endl;
        std::cout << "   Hidden size: " << metadata.hidden_size << std::endl;
        std::cout << "   Layers: " << metadata.num_layers << std::endl;
        std::cout << "   Tensors: " << model_data.tensor_names().size() << std::endl;
        
        if (model_data.tensor_names().size() > 0) {
            std::cout << "   ✅ Model loaded with tensors" << std::endl;
        } else {
            std::cout << "   ❌ No tensors loaded" << std::endl;
            return 1;
        }
        
        // Test 3: Tensor access
        std::cout << "\n3. Testing tensor access..." << std::endl;
        
        const auto& tensor_names = model_data.tensor_names();
        if (!tensor_names.empty()) {
            const auto* first_tensor = model_data.get_tensor(tensor_names[0]);
            std::cout << "   First tensor: " << tensor_names[0] << std::endl;
            std::cout << "   Shape: [";
            for (size_t i = 0; i < first_tensor->shape().dimensions().size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << first_tensor->shape().dimensions()[i];
            }
            std::cout << "]" << std::endl;
            std::cout << "   ✅ Tensor access working" << std::endl;
        }
        
        // Cleanup
        std::remove("quick_test.pth");
        
        std::cout << "\n🎉 All PyTorch loader tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed: " << e.what() << std::endl;
        std::remove("quick_test.pth");
        return 1;
    }
}
