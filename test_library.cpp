#include "turboinfer/turboinfer.hpp"
#include <iostream>

int main() {
    std::cout << "TurboInfer Build Info: " << turboinfer::build_info() << std::endl;
    
    // Initialize TurboInfer
    if (turboinfer::initialize(true)) {
        std::cout << "TurboInfer initialized successfully!" << std::endl;
        
        // Create a simple tensor
        turboinfer::core::TensorShape shape({2, 3});
        turboinfer::core::Tensor tensor(shape, turboinfer::core::DataType::kFloat32);
        std::cout << "Created tensor with shape: " << tensor.shape().ndim() << " dimensions" << std::endl;
        
        // Shutdown
        turboinfer::shutdown();
    } else {
        std::cerr << "Failed to initialize TurboInfer" << std::endl;
        return 1;
    }
    
    return 0;
}
