#include "turboinfer/turboinfer.hpp"
#include <iostream>

int main() {
    // Initialize the library
    if (!turboinfer::initialize(true)) {
        std::cerr << "Failed to initialize TurboInfer" << std::endl;
        return 1;
    }
    
    // Display build information
    std::cout << "Build Info: " << turboinfer::build_info() << std::endl;
    
    // Create a tensor
    turboinfer::core::TensorShape shape({2, 3});
    turboinfer::core::Tensor tensor(shape, turboinfer::core::DataType::kFloat32);
    
    std::cout << "Created tensor with " << tensor.shape().ndim() << " dimensions" << std::endl;
    std::cout << "Total elements: " << tensor.shape().total_size() << std::endl;
    std::cout << "Memory usage: " << tensor.byte_size() << " bytes" << std::endl;
    
    // Perform tensor operations
    auto slice_start = std::vector<size_t>{0, 0};
    auto slice_end = std::vector<size_t>{1, 2};
    auto sliced = tensor.slice(slice_start, slice_end);
    
    std::cout << "Sliced tensor has " << sliced.shape().ndim() << " dimensions" << std::endl;
    
    // Clean up
    turboinfer::shutdown();
    std::cout << "README example completed successfully!" << std::endl;
    return 0;
}
