#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <cstddef>
#include <vector>

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
    std::vector<std::size_t> slice_start;
    slice_start.push_back(0);
    slice_start.push_back(0);
    
    std::vector<std::size_t> slice_end;
    slice_end.push_back(1);
    slice_end.push_back(2);
    auto sliced = tensor.slice(slice_start, slice_end);
    
    std::cout << "Sliced tensor has " << sliced.shape().ndim() << " dimensions" << std::endl;
    
    // Clean up
    turboinfer::shutdown();
    std::cout << "README example completed successfully!" << std::endl;
    return 0;
}
