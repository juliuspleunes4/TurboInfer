/**
 * @file basic_inference.cpp
 * @brief Basic example demonstrating TurboInfer usage for text generation.
 * @author J.J.G. Pleunes
 */

#include <turboinfer/turboinfer.hpp>
#include <iostream>
#include <string>
#include <exception>

int main(int argc, char* argv[]) {
    // Initialize TurboInfer library
    if (!turboinfer::initialize()) {
        std::cerr << "Failed to initialize TurboInfer library" << std::endl;
        return 1;
    }

    try {
        // Check command line arguments
        if (argc < 2) {
            std::cout << "Usage: " << argv[0] << " <model_path> [prompt]" << std::endl;
            std::cout << "Example: " << argv[0] << " model.gguf \"Hello, world!\"" << std::endl;
            return 1;
        }

        std::string model_path = argv[1];
        std::string prompt = (argc >= 3) ? argv[2] : "The future of artificial intelligence is";

        std::cout << "TurboInfer Basic Inference Example" << std::endl;
        std::cout << "Version: " << turboinfer::version() << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        // Configure inference settings
        turboinfer::InferenceConfig config;
        config.max_sequence_length = 2048;
        config.max_batch_size = 1;
        config.temperature = 0.8f;
        config.top_p = 0.9f;
        config.top_k = 50;
        config.device = turboinfer::core::ComputeDevice::kAuto;

        // Create inference engine
        std::cout << "Loading model..." << std::endl;
        // Create inference engine
        auto engine = turboinfer::model::create_engine(model_path, config);

        // Display model information
        const auto& metadata = engine->model_metadata();
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "  Name: " << metadata.name << std::endl;
        std::cout << "  Architecture: " << metadata.architecture << std::endl;
        std::cout << "  Vocabulary size: " << metadata.vocab_size << std::endl;
        std::cout << "  Hidden size: " << metadata.hidden_size << std::endl;
        std::cout << "  Number of layers: " << metadata.num_layers << std::endl;
        std::cout << "  Number of heads: " << metadata.num_heads << std::endl;
        std::cout << "  Memory usage: " << (engine->memory_usage() / 1024 / 1024) << " MB" << std::endl;
        std::cout << std::endl;

        // Generate text
        std::cout << "Generating text..." << std::endl;
        std::cout << std::string(30, '-') << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        auto result = engine->generate(prompt, 50, true);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Display results
        std::string generated_text = engine->decode(result.tokens);
        
        std::cout << "Input: " << prompt << std::endl;
        std::cout << "Generated: " << generated_text << std::endl;
        std::cout << std::string(30, '-') << std::endl;

        // Performance statistics
        std::cout << "Performance Statistics:" << std::endl;
        std::cout << "  Generated tokens: " << result.tokens.size() << std::endl;
        std::cout << "  Total time: " << result.total_time_ms << " ms" << std::endl;
        std::cout << "  Tokens per second: " << result.tokens_per_second << std::endl;
        std::cout << "  Finished: " << (result.finished ? "Yes" : "No") << std::endl;
        if (!result.stop_reason.empty()) {
            std::cout << "  Stop reason: " << result.stop_reason << std::endl;
        }

        // Display log probabilities if available
        if (!result.logprobs.empty()) {
            std::cout << "  Average log probability: ";
            double avg_logprob = 0.0;
            for (float lp : result.logprobs) {
                avg_logprob += lp;
            }
            avg_logprob /= result.logprobs.size();
            std::cout << avg_logprob << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Engine performance stats:" << std::endl;
        std::cout << engine->performance_stats() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        turboinfer::shutdown();
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        turboinfer::shutdown();
        return 1;
    }

    // Clean up
    turboinfer::shutdown();
    
    std::cout << "Example completed successfully!" << std::endl;
    return 0;
}
