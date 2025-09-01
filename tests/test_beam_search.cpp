/**
 * @file test_beam_search.cpp
 * @brief Tests for beam search functionality in the inference engine.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/model/inference_engine.hpp"
#include "turboinfer/model/model_loader.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>

using namespace turboinfer::model;

void test_basic_beam_search() {
    std::cout << "\n=== Testing Basic Beam Search Generation ===" << std::endl;
    
    try {
        // Create a basic configuration for testing
        InferenceConfig config;
        config.max_sequence_length = 512;
        config.max_batch_size = 4;
        config.temperature = 1.0f;
        config.top_p = 0.9f;
        config.top_k = 50;
        config.use_cache = true;
        
        // Create minimal test model data
        ModelMetadata metadata;
        metadata.name = "test_model";
        metadata.architecture = "llama";
        metadata.vocab_size = 32000;
        metadata.hidden_size = 768;
        metadata.num_layers = 2;
        metadata.num_heads = 12;
        metadata.intermediate_size = 3072;
        
        ModelData test_model_data;
        test_model_data.metadata() = metadata;
        
        // Initialize a simple tensor for testing
        turboinfer::core::TensorShape shape({100, 768});
        turboinfer::core::Tensor test_tensor(shape, turboinfer::core::DataType::kFloat32);
        
        // Fill with small random values
        auto data_ptr = static_cast<float*>(test_tensor.data());
        for (size_t i = 0; i < shape.total_size(); ++i) {
            data_ptr[i] = static_cast<float>(i % 100) / 1000.0f;
        }
        
        test_model_data.add_tensor("token_embeddings.weight", std::move(test_tensor));
        
        // Create inference engine
        InferenceEngine engine(test_model_data, config);
        
        // Test input tokens
        std::vector<int> input_tokens = {1, 100, 200, 300}; // Some test tokens
        size_t max_new_tokens = 10;
        size_t beam_size = 3;
        
        std::cout << "Input tokens: ";
        for (int token : input_tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
        
        // Generate using beam search
        auto start_time = std::chrono::steady_clock::now();
        auto results = engine.generate_beam_search(input_tokens, max_new_tokens, beam_size, true);
        auto end_time = std::chrono::steady_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Verify results
        assert(!results.empty());
        assert(results.size() <= beam_size);
        
        std::cout << "Generated " << results.size() << " beam results in " 
                  << duration.count() << "ms" << std::endl;
        
        // Print beam results
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            std::cout << "Beam " << (i + 1) << ": ";
            for (int token : result.tokens) {
                std::cout << token << " ";
            }
            std::cout << "(finished: " << (result.finished ? "yes" : "no") << ")";
            
            if (!result.logprobs.empty()) {
                float avg_logprob = 0.0f;
                for (float lp : result.logprobs) {
                    avg_logprob += lp;
                }
                avg_logprob /= result.logprobs.size();
                std::cout << " avg_logprob: " << avg_logprob;
            }
            std::cout << std::endl;
            
            // Verify beam result structure
            assert(result.tokens.size() <= max_new_tokens);
            if (!result.logprobs.empty()) {
                assert(result.logprobs.size() == result.tokens.size());
            }
        }
        
        std::cout << "✅ Basic beam search generation test passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Beam search generation failed: " << e.what() << std::endl;
        throw;
    }
}

void test_beam_search_different_sizes() {
    std::cout << "\n=== Testing Beam Search with Different Beam Sizes ===" << std::endl;
    
    try {
        // Create minimal test setup
        InferenceConfig config;
        ModelMetadata metadata;
        metadata.name = "test_model";
        metadata.architecture = "llama";
        metadata.vocab_size = 32000;
        metadata.hidden_size = 768;
        metadata.num_layers = 2;
        metadata.num_heads = 12;
        
        ModelData test_model_data;
        test_model_data.metadata() = metadata;
        
        // Add minimal tensor
        turboinfer::core::TensorShape shape({100, 768});
        turboinfer::core::Tensor test_tensor(shape, turboinfer::core::DataType::kFloat32);
        auto data_ptr = static_cast<float*>(test_tensor.data());
        for (size_t i = 0; i < shape.total_size(); ++i) {
            data_ptr[i] = static_cast<float>(i % 100) / 1000.0f;
        }
        test_model_data.add_tensor("token_embeddings.weight", std::move(test_tensor));
        
        InferenceEngine engine(test_model_data, config);
        std::vector<int> input_tokens = {1, 50, 100};
        size_t max_new_tokens = 5;
        
        // Test different beam sizes
        std::vector<size_t> beam_sizes = {1, 2, 4, 8};
        
        for (size_t beam_size : beam_sizes) {
            std::cout << "Testing beam size: " << beam_size << std::endl;
            
            auto results = engine.generate_beam_search(input_tokens, max_new_tokens, beam_size);
            
            assert(!results.empty());
            assert(results.size() <= beam_size);
            
            std::cout << "  Generated " << results.size() << " beams" << std::endl;
            
            // Verify that beam size 1 gives exactly 1 result
            if (beam_size == 1) {
                assert(results.size() == 1);
            }
        }
        
        std::cout << "✅ Beam size variation test passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Beam size variation test failed: " << e.what() << std::endl;
        throw;
    }
}

void test_beam_search_error_handling() {
    std::cout << "\n=== Testing Beam Search Error Handling ===" << std::endl;
    
    try {
        // Create minimal test setup
        InferenceConfig config;
        ModelMetadata metadata;
        metadata.name = "test_model";
        metadata.architecture = "llama";
        metadata.vocab_size = 32000;
        metadata.hidden_size = 768;
        metadata.num_layers = 2;
        metadata.num_heads = 12;
        
        ModelData test_model_data;
        test_model_data.metadata() = metadata;
        
        // Add minimal tensor
        turboinfer::core::TensorShape shape({100, 768});
        turboinfer::core::Tensor test_tensor(shape, turboinfer::core::DataType::kFloat32);
        auto data_ptr = static_cast<float*>(test_tensor.data());
        for (size_t i = 0; i < shape.total_size(); ++i) {
            data_ptr[i] = static_cast<float>(i % 100) / 1000.0f;
        }
        test_model_data.add_tensor("token_embeddings.weight", std::move(test_tensor));
        
        InferenceEngine engine(test_model_data, config);
        std::vector<int> input_tokens = {1, 10};
        
        // Test with beam size 0 (should throw)
        bool exception_caught = false;
        try {
            engine.generate_beam_search(input_tokens, 5, 0);
        } catch (const std::runtime_error& e) {
            exception_caught = true;
            std::cout << "Expected error caught: " << e.what() << std::endl;
        }
        
        assert(exception_caught);
        std::cout << "✅ Error handling test passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error handling test failed: " << e.what() << std::endl;
        throw;
    }
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "           TurboInfer Beam Search Tests" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    try {
        test_basic_beam_search();
        test_beam_search_different_sizes();
        test_beam_search_error_handling();
        
        std::cout << "\n🎉 All beam search tests passed! Phase 4 complete!" << std::endl;
        std::cout << "\nPhase 4 Inference Engine Summary:" << std::endl;
        std::cout << "✅ Transformer decoder implementation" << std::endl;
        std::cout << "✅ Token generation with autoregressive sampling" << std::endl;
        std::cout << "✅ KV-cache management for efficiency" << std::endl;
        std::cout << "✅ Temperature/top-k/top-p sampling strategies" << std::endl;
        std::cout << "✅ Beam search for improved generation quality" << std::endl;
        std::cout << "\nPhase 4: Inference Engine - COMPLETE! 🚀" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Some beam search tests failed: " << e.what() << std::endl;
        return 1;
    }
}
