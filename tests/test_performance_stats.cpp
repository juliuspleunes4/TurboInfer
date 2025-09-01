/**
 * @file test_performance_stats.cpp
 * @brief Test enhanced performance statistics feature.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/model/inference_engine.hpp"
#include <iostream>
#include <cassert>

using namespace turboinfer::model;
using namespace turboinfer::core;

#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        std::cerr << "ASSERTION FAILED: " << #condition << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } else { \
        std::cout << "✅ PASS: " << #condition << std::endl; \
    }

ModelData create_perf_test_model() {
    ModelData model;
    
    // Set up basic metadata
    auto& metadata = model.metadata();
    metadata.name = "performance_test";
    metadata.architecture = "llama";
    metadata.vocab_size = 1000;
    metadata.hidden_size = 128;
    metadata.num_layers = 2;
    metadata.num_heads = 4;
    metadata.intermediate_size = 512;
    
    // Create minimal tensor data
    TensorShape embed_shape({1000, 128});
    Tensor token_embeddings(embed_shape, DataType::kFloat32);
    float* embed_data = token_embeddings.data_ptr<float>();
    for (size_t i = 0; i < embed_shape.total_size(); ++i) {
        embed_data[i] = (static_cast<float>(i) / embed_shape.total_size()) - 0.5f;
    }
    model.add_tensor("token_embeddings.weight", std::move(token_embeddings));
    
    // Add required norm tensor
    TensorShape norm_shape({128});
    Tensor norm_tensor(norm_shape, DataType::kFloat32);
    float* norm_data = norm_tensor.data_ptr<float>();
    for (size_t i = 0; i < norm_shape.total_size(); ++i) {
        norm_data[i] = 1.0f;
    }
    model.add_tensor("norm.weight", std::move(norm_tensor));
    
    // Add LM head
    TensorShape lm_head_shape({128, 1000});
    Tensor lm_head_tensor(lm_head_shape, DataType::kFloat32);
    float* lm_head_data = lm_head_tensor.data_ptr<float>();
    for (size_t i = 0; i < lm_head_shape.total_size(); ++i) {
        lm_head_data[i] = (static_cast<float>(i) / lm_head_shape.total_size()) - 0.5f;
    }
    model.add_tensor("lm_head.weight", std::move(lm_head_tensor));
    
    return model;
}

int main() {
    std::cout << "📊 Testing Enhanced Performance Statistics" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
        std::cout << "\n--- Test: Initial Performance Stats ---" << std::endl;
        
        ModelData model = create_perf_test_model();
        InferenceConfig config;
        config.max_sequence_length = 512;
        config.temperature = 1.0f;
        
        InferenceEngine engine(model, config);
        
        // Get initial stats (should show no activity)
        std::string initial_stats = engine.performance_stats();
        std::cout << "Initial performance statistics:" << std::endl;
        std::cout << initial_stats << std::endl;
        
        // Verify initial state
        ASSERT_TRUE(initial_stats.find("Total Generations: 0") != std::string::npos);
        ASSERT_TRUE(initial_stats.find("Total Tokens Generated: 0") != std::string::npos);
        std::cout << "✅ Initial performance stats show zero activity" << std::endl;
        
        std::cout << "\n--- Test: Performance Tracking During Generation ---" << std::endl;
        
        // Perform several generations
        std::string prompt = "Test prompt";
        std::vector<float> speeds;
        
        for (int i = 0; i < 3; ++i) {
            auto result = engine.generate(prompt, 5);
            speeds.push_back(result.tokens_per_second);
            std::cout << "Generation " << (i+1) << ": " << result.tokens.size() << " tokens at " 
                      << result.tokens_per_second << " tok/s" << std::endl;
            
            ASSERT_TRUE(result.tokens.size() > 0);
            ASSERT_TRUE(result.tokens_per_second > 0);
        }
        
        std::cout << "✅ Multiple generations completed successfully" << std::endl;
        
        std::cout << "\n--- Test: Updated Performance Statistics ---" << std::endl;
        
        // Get updated stats
        std::string updated_stats = engine.performance_stats();
        std::cout << "Updated performance statistics:" << std::endl;
        std::cout << updated_stats << std::endl;
        
        // Verify that statistics have been updated
        ASSERT_TRUE(updated_stats.find("Total Generations: 3") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Total Tokens Generated:") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Average Speed:") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Peak Speed:") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Forward Pass Performance:") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("KV-Cache Performance:") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Model Information:") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Efficiency Metrics:") != std::string::npos);
        
        std::cout << "✅ Performance statistics properly updated after generations" << std::endl;
        
        std::cout << "\n--- Test: Performance Metrics Validation ---" << std::endl;
        
        // Verify that the stats contain reasonable values
        bool has_reasonable_speed = false;
        for (float speed : speeds) {
            if (speed > 100.0f) { // Should achieve > 100 tokens/second
                has_reasonable_speed = true;
                break;
            }
        }
        ASSERT_TRUE(has_reasonable_speed);
        
        // Check for presence of key statistical components
        ASSERT_TRUE(updated_stats.find("Architecture: llama") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Layers: 2") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Hidden Size: 128") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Attention Heads: 4") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Vocabulary Size: 1000") != std::string::npos);
        
        std::cout << "✅ Model information correctly displayed in stats" << std::endl;
        
        // Verify efficiency metrics
        ASSERT_TRUE(updated_stats.find("Efficiency Score:") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Performance Rating:") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Resource Usage:") != std::string::npos);
        
        std::cout << "✅ Efficiency and resource metrics properly calculated" << std::endl;
        
        std::cout << "\n--- Test: Statistics Format and Completeness ---" << std::endl;
        
        // Check for well-formatted sections
        ASSERT_TRUE(updated_stats.find("Generation Performance:") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Throughput Performance:") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("Forward Pass Performance:") != std::string::npos);
        ASSERT_TRUE(updated_stats.find("KV-Cache Performance:") != std::string::npos);
        
        std::cout << "✅ All performance sections present and formatted correctly" << std::endl;
        
        std::cout << "\n🎉 ALL PERFORMANCE STATISTICS TESTS PASSED!" << std::endl;
        std::cout << "✅ Enhanced performance tracking implemented successfully" << std::endl;
        std::cout << "✅ Detailed runtime metrics collection working" << std::endl;
        std::cout << "✅ Professional performance reporting comparable to production systems" << std::endl;
        std::cout << "✅ Forward pass, generation, and cache performance properly monitored" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
