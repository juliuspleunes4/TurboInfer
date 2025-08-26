/**
 * @file test_advanced_math.cpp
 * @brief Test program for advanced mathematical operations (attention, multi-head attention, RoPE)
 * @author J.J.G. Pleunes
 */

#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace turboinfer;
using namespace turboinfer::core;

void print_tensor_detailed(const std::string& name, const Tensor& tensor) {
    std::cout << name << " (shape: [";
    const auto& dims = tensor.shape().dimensions();
    for (size_t i = 0; i < dims.size(); ++i) {
        std::cout << dims[i];
        if (i < dims.size() - 1) std::cout << ", ";
    }
    std::cout << "]):" << std::endl;
    
    if (tensor.dtype() == DataType::kFloat32) {
        const float* data = tensor.data_ptr<float>();
        size_t total = tensor.shape().total_size();
        
        // Print in a nice matrix format for 3D tensors
        if (dims.size() == 3) {
            size_t batch = dims[0];
            size_t seq = dims[1];
            size_t hidden = dims[2];
            
            for (size_t b = 0; b < batch && b < 2; ++b) {  // Limit to first 2 batches
                std::cout << "  Batch " << b << ":" << std::endl;
                for (size_t s = 0; s < seq && s < 4; ++s) {  // Limit to first 4 sequences
                    std::cout << "    [";
                    for (size_t h = 0; h < hidden && h < 8; ++h) {  // Limit to first 8 features
                        size_t idx = b * seq * hidden + s * hidden + h;
                        std::cout << std::fixed << std::setprecision(3) << data[idx];
                        if (h < std::min(hidden, size_t(8)) - 1) std::cout << ", ";
                    }
                    if (hidden > 8) std::cout << ", ...";
                    std::cout << "]" << std::endl;
                }
                if (seq > 4) std::cout << "    ..." << std::endl;
            }
            if (batch > 2) std::cout << "  ..." << std::endl;
        } else {
            // Simple format for other dimensions
            std::cout << "  [";
            for (size_t i = 0; i < std::min(total, size_t(16)); ++i) {
                std::cout << std::fixed << std::setprecision(3) << data[i];
                if (i < std::min(total, size_t(16)) - 1) std::cout << ", ";
            }
            if (total > 16) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== TurboInfer Advanced Mathematical Operations Test ===" << std::endl;
    
    // Initialize the library
    if (!initialize()) {
        std::cerr << "Failed to initialize TurboInfer!" << std::endl;
        return 1;
    }
    
    try {
        // Create tensor engine
        TensorEngine engine(ComputeDevice::kCPU);
        
        std::cout << "1. Testing Single-Head Attention:" << std::endl;
        
        // Create query, key, value tensors [batch=1, seq_len=4, hidden=8]
        Tensor query(TensorShape({1, 4, 8}), DataType::kFloat32);
        Tensor key(TensorShape({1, 4, 8}), DataType::kFloat32);
        Tensor value(TensorShape({1, 4, 8}), DataType::kFloat32);
        
        // Fill with test data
        float* q_data = query.data_ptr<float>();
        float* k_data = key.data_ptr<float>();
        float* v_data = value.data_ptr<float>();
        
        // Simple pattern for testing
        for (int i = 0; i < 32; ++i) {  // 1*4*8 = 32 elements
            q_data[i] = static_cast<float>(i + 1) * 0.1f;
            k_data[i] = static_cast<float>(i + 2) * 0.1f;
            v_data[i] = static_cast<float>(i + 3) * 0.1f;
        }
        
        print_tensor_detailed("Query", query);
        print_tensor_detailed("Key", key);
        print_tensor_detailed("Value", value);
        
        // Test attention
        Tensor attention_output = engine.attention(query, key, value, nullptr);
        print_tensor_detailed("Attention Output", attention_output);
        
        std::cout << "2. Testing Multi-Head Attention:" << std::endl;
        
        // Create larger tensors for multi-head attention [batch=1, seq_len=3, hidden=12]
        Tensor mh_query(TensorShape({1, 3, 12}), DataType::kFloat32);
        Tensor mh_key(TensorShape({1, 3, 12}), DataType::kFloat32);
        Tensor mh_value(TensorShape({1, 3, 12}), DataType::kFloat32);
        
        float* mhq_data = mh_query.data_ptr<float>();
        float* mhk_data = mh_key.data_ptr<float>();
        float* mhv_data = mh_value.data_ptr<float>();
        
        // Fill with test data
        for (int i = 0; i < 36; ++i) {  // 1*3*12 = 36 elements
            mhq_data[i] = static_cast<float>(i + 1) * 0.05f;
            mhk_data[i] = static_cast<float>(i + 5) * 0.05f;
            mhv_data[i] = static_cast<float>(i + 10) * 0.05f;
        }
        
        print_tensor_detailed("Multi-Head Query", mh_query);
        
        // Test multi-head attention with 3 heads (12 hidden / 3 heads = 4 dimensions per head)
        Tensor mh_output = engine.multi_head_attention(mh_query, mh_key, mh_value, 3, nullptr);
        print_tensor_detailed("Multi-Head Attention Output", mh_output);
        
        std::cout << "3. Testing RoPE (Rotary Position Embedding):" << std::endl;
        
        // Create input tensor [batch=1, seq_len=4, hidden=8] (hidden must be even for RoPE)
        Tensor rope_input(TensorShape({1, 4, 8}), DataType::kFloat32);
        Tensor position_ids(TensorShape({4}), DataType::kFloat32);
        
        float* rope_data = rope_input.data_ptr<float>();
        float* pos_data = position_ids.data_ptr<float>();
        
        // Fill with test data
        for (int i = 0; i < 32; ++i) {
            rope_data[i] = static_cast<float>(i + 1) * 0.1f;
        }
        
        // Position IDs: [0, 1, 2, 3]
        for (int i = 0; i < 4; ++i) {
            pos_data[i] = static_cast<float>(i);
        }
        
        print_tensor_detailed("RoPE Input", rope_input);
        print_tensor_detailed("Position IDs", position_ids);
        
        // Test RoPE with theta=10000 (common value)
        Tensor rope_output = engine.apply_rope(rope_input, position_ids, 10000.0f);
        print_tensor_detailed("RoPE Output", rope_output);
        
        std::cout << "4. Testing 4D RoPE (Multi-Head format):" << std::endl;
        
        // Create 4D input tensor [batch=1, num_heads=2, seq_len=3, head_dim=4]
        Tensor rope_4d_input(TensorShape({1, 2, 3, 4}), DataType::kFloat32);
        Tensor position_ids_3(TensorShape({3}), DataType::kFloat32);
        
        float* rope_4d_data = rope_4d_input.data_ptr<float>();
        float* pos_3_data = position_ids_3.data_ptr<float>();
        
        // Fill with test data
        for (int i = 0; i < 24; ++i) {  // 1*2*3*4 = 24 elements
            rope_4d_data[i] = static_cast<float>(i + 1) * 0.1f;
        }
        
        for (int i = 0; i < 3; ++i) {
            pos_3_data[i] = static_cast<float>(i);
        }
        
        print_tensor_detailed("RoPE 4D Input", rope_4d_input);
        
        Tensor rope_4d_output = engine.apply_rope(rope_4d_input, position_ids_3, 10000.0f);
        print_tensor_detailed("RoPE 4D Output", rope_4d_output);
        
        std::cout << "=== All Advanced Operations Completed Successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        shutdown();
        return 1;
    }
    
    shutdown();
    return 0;
}
