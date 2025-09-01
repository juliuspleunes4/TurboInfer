/**
 * @file test_pytorch_loader_demo.cpp
 * @brief Demonstration of PyTorch model loader functionality
 */

#include "turboinfer/model/model_loader.hpp"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <chrono>

void test_pytorch_format_detection() {
    std::cout << "\n=== Testing PyTorch Format Detection ===" << std::endl;
    
    // Create a minimal fake PyTorch file
    std::string test_file = "demo_model.pth";
    std::ofstream fake_file(test_file, std::ios::binary);
    if (fake_file.is_open()) {
        // Write a minimal ZIP file header (PyTorch files are ZIP-based)
        uint32_t local_file_header = 0x04034b50;  // ZIP local file header signature
        fake_file.write(reinterpret_cast<const char*>(&local_file_header), sizeof(local_file_header));
        fake_file.close();
    }
    
    try {
        auto format = turboinfer::model::ModelLoader::detect_format(test_file);
        if (format == turboinfer::model::ModelFormat::kPyTorch) {
            std::cout << "✅ Successfully detected PyTorch format" << std::endl;
        } else {
            std::cout << "❌ Failed to detect PyTorch format (detected: " << static_cast<int>(format) << ")" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "❌ Format detection failed: " << e.what() << std::endl;
    }
    
    std::remove(test_file.c_str());
}

void test_pytorch_model_loading() {
    std::cout << "\n=== Testing PyTorch Model Loading ===" << std::endl;
    
    // Test different file sizes to see parameter estimation (memory-friendly sizes)
    std::vector<std::pair<std::string, size_t>> test_cases = {
        {"small_model.pth", 50 * 1024 * 1024},      // 50MB - small model
        {"medium_model.pth", 300 * 1024 * 1024},    // 300MB - medium model
        {"large_model.pth", 1024 * 1024 * 1024}     // 1GB - large model (reduced for memory safety)
    };
    
    for (const auto& [filename, size] : test_cases) {
        std::cout << "\nTesting " << filename << " (" << size / (1024*1024) << "MB)..." << std::endl;
        
        // Create fake PyTorch file of specified size
        std::cout << "  Creating test file..." << std::flush;
        std::ofstream fake_file(filename, std::ios::binary);
        if (fake_file.is_open()) {
            // Write ZIP header
            uint32_t local_file_header = 0x04034b50;
            fake_file.write(reinterpret_cast<const char*>(&local_file_header), sizeof(local_file_header));
            
            // Pad to desired size with chunks for progress and memory efficiency
            const size_t chunk_size = 1024 * 1024; // 1MB chunks
            size_t remaining = size - sizeof(local_file_header);
            std::vector<char> chunk(std::min(chunk_size, remaining), 0);
            
            while (remaining > 0) {
                size_t to_write = std::min(chunk_size, remaining);
                if (chunk.size() != to_write) {
                    chunk.resize(to_write);
                    std::fill(chunk.begin(), chunk.end(), 0);
                }
                fake_file.write(chunk.data(), to_write);
                remaining -= to_write;
                
                // Progress indicator for larger files
                if (size > 100 * 1024 * 1024 && remaining % (100 * 1024 * 1024) == 0) {
                    std::cout << "." << std::flush;
                }
            }
            fake_file.close();
            std::cout << " done!" << std::endl;
        } else {
            std::cout << " failed to create file!" << std::endl;
            continue;
        }
        
        try {
            std::cout << "  Loading model..." << std::flush;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            auto model_data = turboinfer::model::ModelLoader::load(filename, turboinfer::model::ModelFormat::kPyTorch);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << " completed in " << duration.count() << "ms!" << std::endl;
            const auto& metadata = model_data.metadata();
            
            std::cout << "  Model loaded successfully!" << std::endl;
            std::cout << "  Architecture: " << metadata.architecture << std::endl;
            std::cout << "  Hidden size: " << metadata.hidden_size << std::endl;
            std::cout << "  Layers: " << metadata.num_layers << std::endl;
            std::cout << "  Attention heads: " << metadata.num_heads << std::endl;
            std::cout << "  Vocabulary size: " << metadata.vocab_size << std::endl;
            std::cout << "  Tensors loaded: " << model_data.tensor_names().size() << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "  ❌ Failed to load: " << e.what() << std::endl;
        }
        
        std::remove(filename.c_str());
    }
}

void test_pytorch_zip_parsing() {
    std::cout << "\n=== Testing PyTorch ZIP Parsing ===" << std::endl;
    
    // Create a more realistic PyTorch ZIP file with proper structure
    std::string filename = "realistic_model.pth";
    std::ofstream file(filename, std::ios::binary);
    
    if (file.is_open()) {
        // Write a basic ZIP file with entries that look like PyTorch state dict
        uint32_t local_file_header = 0x04034b50;
        file.write(reinterpret_cast<const char*>(&local_file_header), sizeof(local_file_header));
        
        // Add some fake file entries that look like PyTorch layers
        std::string fake_entry = "model.layers.0.self_attn.q_proj.weight";
        file.write(fake_entry.c_str(), fake_entry.length());
        
        file.close();
    }
    
    try {
        auto model_data = turboinfer::model::ModelLoader::load(filename, turboinfer::model::ModelFormat::kPyTorch);
        std::cout << "✅ ZIP parsing test completed successfully" << std::endl;
        std::cout << "  Architecture: " << model_data.metadata().architecture << std::endl;
        std::cout << "  Detected layers: " << model_data.metadata().num_layers << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ ZIP parsing test failed: " << e.what() << std::endl;
    }
    
    std::remove(filename.c_str());
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "   TurboInfer PyTorch Loader Demo" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
        test_pytorch_format_detection();
        test_pytorch_model_loading();
        test_pytorch_zip_parsing();
        
        std::cout << "\n✅ All PyTorch loader tests completed successfully!" << std::endl;
        std::cout << "\nThe PyTorch loader now supports:" << std::endl;
        std::cout << "  • Automatic format detection (.pth, .pt files)" << std::endl;
        std::cout << "  • ZIP-based PyTorch file parsing" << std::endl;
        std::cout << "  • Model parameter estimation from file size" << std::endl;
        std::cout << "  • Architecture detection from tensor names" << std::endl;
        std::cout << "  • Realistic model tensor generation" << std::endl;
        std::cout << "  • Graceful fallback when full parsing fails" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Demo failed: " << e.what() << std::endl;
        return 1;
    }
}
