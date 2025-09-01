#include "turboinfer/turboinfer.hpp"
#include <iostream>

int main() {
    std::cout << "🚀 Testing Enhanced TensorEngine Features..." << std::endl;
    
    // Initialize TurboInfer
    if (!turboinfer::initialize()) {
        std::cerr << "Failed to initialize TurboInfer" << std::endl;
        return 1;
    }
    
    try {
        // Create TensorEngine
        turboinfer::core::TensorEngine engine;
        
        // Test GPU availability detection
        std::cout << "\n--- GPU Availability Detection ---" << std::endl;
        bool gpu_available = engine.gpu_available();
        std::cout << "GPU Available: " << (gpu_available ? "Yes" : "No") << std::endl;
        
        // Test enhanced device info
        std::cout << "\n--- Enhanced Device Information ---" << std::endl;
        std::string device_info = engine.device_info();
        std::cout << device_info << std::endl;
        
        std::cout << "\n🎉 Enhanced TensorEngine features test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        turboinfer::shutdown();
        return 1;
    }
    
    turboinfer::shutdown();
    return 0;
}
