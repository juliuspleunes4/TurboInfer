/**
 * @file turboinfer.hpp
 * @brief Main header file for the TurboInfer library.
 * @author J.J.G. Pleunes
 * 
 * This header provides access to all public APIs of the TurboInfer library.
 * Include this file to use TurboInfer in your applications.
 */

#pragma once

// Core functionality
#include "core/tensor.hpp"
#include "core/tensor_engine.hpp"

// Model handling
#include "model/model_loader.hpp"
#include "model/inference_engine.hpp"

// Optimization utilities
#include "optimize/quantization.hpp"

// Utility functions
#include "util/logging.hpp"
#include "util/profiler.hpp"

/**
 * @namespace turboinfer
 * @brief Main namespace for the TurboInfer library.
 */
namespace turboinfer {

/**
 * @brief Library version information.
 */
struct Version {
    static constexpr int kMajor = 1;        ///< Major version number
    static constexpr int kMinor = 0;        ///< Minor version number
    static constexpr int kPatch = 0;        ///< Patch version number
    static constexpr const char* kString = "1.0.0"; ///< Full version string
};

/**
 * @brief Gets the library version as a string.
 * @return Version string in format "major.minor.patch".
 */
inline const char* version() {
    return Version::kString;
}

/**
 * @brief Gets build information.
 * @return Build information string.
 */
const char* build_info();

/**
 * @brief Initializes the TurboInfer library.
 * 
 * This function should be called once before using any TurboInfer functionality.
 * It initializes internal systems, detects available hardware, and sets up
 * optimal default configurations.
 * 
 * @param enable_logging Whether to enable logging output.
 * @return True if initialization succeeded, false otherwise.
 */
bool initialize(bool enable_logging = true);

/**
 * @brief Shuts down the TurboInfer library.
 * 
 * This function cleans up resources and should be called before application exit.
 * After calling this function, no other TurboInfer functions should be used.
 */
void shutdown();

/**
 * @brief Checks if the library has been initialized.
 * @return True if initialized, false otherwise.
 */
bool is_initialized();

// Convenience aliases for commonly used types
using Tensor = core::Tensor;
using TensorShape = core::TensorShape;
using TensorEngine = core::TensorEngine;
using ModelData = model::ModelData;
using ModelLoader = model::ModelLoader;
using InferenceEngine = model::InferenceEngine;
using InferenceConfig = model::InferenceConfig;
using GenerationResult = model::GenerationResult;

// Convenience functions for common operations

/**
 * @brief Loads a model from a file.
 * @param file_path Path to the model file.
 * @return Loaded model data.
 * @throws std::runtime_error if loading fails.
 */
inline ModelData load_model(const std::string& file_path) {
    return ModelLoader::load(file_path);
}

/**
 * @brief Tokenizes text into token IDs.
 * @param text Input text to tokenize.
 * @param model_path Path to the model file (for tokenizer).
 * @return Vector of token IDs.
 * @throws std::runtime_error if tokenization fails.
 */
std::vector<int> tokenize(const std::string& text, const std::string& model_path);

/**
 * @brief Detokenizes token IDs back to text.
 * @param tokens Vector of token IDs.
 * @param model_path Path to the model file (for tokenizer).
 * @return Decoded text.
 * @throws std::runtime_error if detokenization fails.
 */
std::string detokenize(const std::vector<int>& tokens, const std::string& model_path);

/**
 * @brief Simple text generation function.
 * @param model_path Path to the model file.
 * @param prompt Input prompt.
 * @param max_tokens Maximum number of tokens to generate.
 * @param temperature Sampling temperature.
 * @return Generated text.
 * @throws std::runtime_error if generation fails.
 */
inline std::string generate_text(const std::string& model_path,
                                const std::string& prompt,
                                size_t max_tokens = 50,
                                float temperature = 1.0f) {
    return model::quick_generate(model_path, prompt, max_tokens, temperature);
}

} // namespace turboinfer
