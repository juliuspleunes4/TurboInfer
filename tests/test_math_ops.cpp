/**
 * @file test_math_ops.cpp
 * @brief Simple test program to demonstrate TurboInfer's mathematical operations
 * @author J.J.G. Pleunes
 */

#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <vector>

using namespace turboinfer;
using namespace turboinfer::core;

void print_tensor(const std::string& name, const Tensor& tensor) {
    std::cout << name << " (shape: [";
    const auto& dims = tensor.shape().dimensions();
    for (size_t i = 0; i < dims.size(); ++i) {
        std::cout << dims[i];
        if (i < dims.size() - 1) std::cout << ", ";
    }
    std::cout << "]): ";
    
    if (tensor.dtype() == DataType::kFloat32) {
        const float* data = tensor.data_ptr<float>();
        size_t total = tensor.shape().total_size();
        for (size_t i = 0; i < std::min(total, size_t(10)); ++i) {
            std::cout << data[i] << " ";
        }
        if (total > 10) std::cout << "...";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== TurboInfer Mathematical Operations Demo ===" << std::endl;
    
    // Initialize the library
    if (!initialize()) {
        std::cerr << "Failed to initialize TurboInfer!" << std::endl;
        return 1;
    }
    
    try {
        // Create tensor engine
        TensorEngine engine(ComputeDevice::kCPU);
        
        std::cout << "\n1. Testing Basic Tensor Operations:" << std::endl;
        
        // Create test tensors
        Tensor a(TensorShape({2, 3}), DataType::kFloat32);
        Tensor b(TensorShape({2, 3}), DataType::kFloat32);
        
        // Fill with test data
        float* a_data = a.data_ptr<float>();
        float* b_data = b.data_ptr<float>();
        
        for (int i = 0; i < 6; ++i) {
            a_data[i] = static_cast<float>(i + 1);  // [1, 2, 3, 4, 5, 6]
            b_data[i] = static_cast<float>(i * 0.5f + 0.5f);  // [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        }
        
        print_tensor("Tensor A", a);
        print_tensor("Tensor B", b);
        
        // Test element-wise addition
        Tensor sum = engine.add(a, b);
        print_tensor("A + B", sum);
        
        // Test element-wise multiplication
        Tensor product = engine.multiply(a, b);
        print_tensor("A * B", product);
        
        // Test scaling
        Tensor scaled = engine.scale(a, 2.0f);
        print_tensor("A * 2.0", scaled);
        
        std::cout << "\n2. Testing Activation Functions:" << std::endl;
        
        // Create input for activations
        Tensor input(TensorShape({1, 4}), DataType::kFloat32);
        float* input_data = input.data_ptr<float>();
        input_data[0] = -2.0f;
        input_data[1] = -0.5f;
        input_data[2] = 0.5f;
        input_data[3] = 2.0f;
        
        print_tensor("Input", input);
        
        // Test ReLU
        Tensor relu_result = engine.relu(input);
        print_tensor("ReLU(input)", relu_result);
        
        // Test GELU
        Tensor gelu_result = engine.gelu(input);
        print_tensor("GELU(input)", gelu_result);
        
        // Test SiLU
        Tensor silu_result = engine.silu(input);
        print_tensor("SiLU(input)", silu_result);
        
        // Test Softmax
        Tensor softmax_result = engine.softmax(input, 1.0f);
        print_tensor("Softmax(input)", softmax_result);
        
        std::cout << "\n3. Testing Matrix Operations:" << std::endl;
        
        // Create matrices for multiplication
        Tensor matrix_a(TensorShape({2, 3}), DataType::kFloat32);
        Tensor matrix_b(TensorShape({3, 2}), DataType::kFloat32);
        
        float* ma_data = matrix_a.data_ptr<float>();
        float* mb_data = matrix_b.data_ptr<float>();
        
        // Fill matrix A: [[1, 2, 3], [4, 5, 6]]
        for (int i = 0; i < 6; ++i) {
            ma_data[i] = static_cast<float>(i + 1);
        }
        
        // Fill matrix B: [[1, 2], [3, 4], [5, 6]]
        for (int i = 0; i < 6; ++i) {
            mb_data[i] = static_cast<float>(i + 1);
        }
        
        print_tensor("Matrix A", matrix_a);
        print_tensor("Matrix B", matrix_b);
        
        // Test matrix multiplication
        Tensor matmul_result = engine.matmul(matrix_a, matrix_b);
        print_tensor("A @ B", matmul_result);
        
        std::cout << "\n4. Testing Bias Addition:" << std::endl;
        
        // Create bias vector
        Tensor bias(TensorShape({2}), DataType::kFloat32);
        float* bias_data = bias.data_ptr<float>();
        bias_data[0] = 10.0f;
        bias_data[1] = 20.0f;
        
        print_tensor("Bias", bias);
        print_tensor("Matrix Result", matmul_result);
        
        // Test bias addition
        Tensor biased_result = engine.add_bias(matmul_result, bias);
        print_tensor("Result + Bias", biased_result);
        
        std::cout << "\n=== All Operations Completed Successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        shutdown();
        return 1;
    }
    
    shutdown();
    return 0;
}
