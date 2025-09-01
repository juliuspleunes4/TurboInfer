/**
 * @file inference_engine.hpp
 * @brief Defines the InferenceEngine class for running LLM inference.
 * @author J.J.G. Pleunes
 */

#pragma once

#include "../core/tensor.hpp"
#include "../core/tensor_engine.hpp"
#include "model_loader.hpp"
#include <vector>
#include <string>
#include <memory>
#include <cstddef>
#include <unordered_map>

namespace turboinfer {
namespace model {

/**
 * @struct InferenceConfig
 * @brief Configuration parameters for inference.
 */
struct InferenceConfig {
    size_t max_sequence_length = 2048;     ///< Maximum sequence length
    size_t max_batch_size = 32;            ///< Maximum batch size
    float temperature = 1.0f;              ///< Sampling temperature
    float top_p = 0.9f;                    ///< Top-p sampling threshold
    size_t top_k = 50;                     ///< Top-k sampling limit
    float length_penalty = 1.0f;           ///< Length penalty for beam search
    int eos_token_id = 2;                  ///< End-of-sequence token ID
    bool use_cache = true;                 ///< Enable KV cache for efficiency
    core::ComputeDevice device = core::ComputeDevice::kAuto; ///< Compute device
};

/**
 * @struct GenerationResult
 * @brief Contains the result of text generation.
 */
struct GenerationResult {
    std::vector<int> tokens;               ///< Generated token IDs
    std::vector<float> logprobs;           ///< Log probabilities (if requested)
    float total_time_ms;                   ///< Total generation time in milliseconds
    float tokens_per_second;               ///< Generation speed in tokens/second
    bool finished;                         ///< Whether generation completed normally
    std::string stop_reason;               ///< Reason for stopping generation
};

/**
 * @class InferenceEngine
 * @brief High-level interface for running inference on language models.
 * 
 * The InferenceEngine provides a simple API for text generation using pre-trained
 * language models. It handles tokenization, batching, KV caching, and sampling
 * strategies internally.
 */
class InferenceEngine {
public:
    /**
     * @brief Constructs an inference engine with a loaded model.
     * @param model_data Loaded model data.
     * @param config Inference configuration.
     */
    explicit InferenceEngine(const ModelData& model_data, 
                           const InferenceConfig& config = InferenceConfig{});

    /**
     * @brief Constructs an inference engine by loading a model from file.
     * @param model_path Path to the model file.
     * @param config Inference configuration.
     */
    explicit InferenceEngine(const std::string& model_path,
                           const InferenceConfig& config = InferenceConfig{});

    /**
     * @brief Destructor.
     */
    ~InferenceEngine();

    // Non-copyable but movable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) noexcept;
    InferenceEngine& operator=(InferenceEngine&&) noexcept;

    /**
     * @brief Gets the model metadata.
     * @return Model metadata.
     */
    const ModelMetadata& model_metadata() const noexcept { return model_metadata_; }

    /**
     * @brief Gets the inference configuration.
     * @return Inference configuration.
     */
    const InferenceConfig& config() const noexcept { return config_; }

    /**
     * @brief Updates the inference configuration.
     * @param config New configuration.
     */
    void set_config(const InferenceConfig& config);

    /**
     * @brief Generates text from a prompt.
     * @param prompt Input text prompt.
     * @param max_new_tokens Maximum number of tokens to generate.
     * @param include_logprobs Whether to include log probabilities in result.
     * @return Generation result.
     * @throws std::runtime_error if generation fails.
     */
    GenerationResult generate(const std::string& prompt, 
                             size_t max_new_tokens,
                             bool include_logprobs = false);

    /**
     * @brief Generates text from token IDs.
     * @param input_tokens Input token sequence.
     * @param max_new_tokens Maximum number of tokens to generate.
     * @param include_logprobs Whether to include log probabilities in result.
     * @return Generation result.
     * @throws std::runtime_error if generation fails.
     */
    GenerationResult generate(const std::vector<int>& input_tokens,
                             size_t max_new_tokens,
                             bool include_logprobs = false);

    /**
     * @brief Generates text for multiple prompts in a batch.
     * @param prompts Vector of input prompts.
     * @param max_new_tokens Maximum number of tokens to generate per prompt.
     * @param include_logprobs Whether to include log probabilities in results.
     * @return Vector of generation results.
     * @throws std::runtime_error if batch generation fails.
     */
    std::vector<GenerationResult> generate_batch(const std::vector<std::string>& prompts,
                                                 size_t max_new_tokens,
                                                 bool include_logprobs = false);

    /**
     * @brief Generates text for multiple token sequences in a batch.
     * @param input_token_batches Vector of input token sequences.
     * @param max_new_tokens Maximum number of tokens to generate per sequence.
     * @param include_logprobs Whether to include log probabilities in results.
     * @return Vector of generation results.
     * @throws std::runtime_error if batch generation fails.
     */
    std::vector<GenerationResult> generate_batch(
        const std::vector<std::vector<int>>& input_token_batches,
        size_t max_new_tokens,
        bool include_logprobs = false);

    /**
     * @brief Generates text using beam search for better quality results.
     * @param input_tokens Input token sequence.
     * @param max_new_tokens Maximum number of tokens to generate.
     * @param beam_size Number of beams to maintain during search.
     * @param include_logprobs Whether to include log probabilities in results.
     * @return Vector of generation results (one per beam, sorted by score).
     * @throws std::runtime_error if beam search fails.
     */
    std::vector<GenerationResult> generate_beam_search(
        const std::vector<int>& input_tokens,
        size_t max_new_tokens,
        size_t beam_size = 4,
        bool include_logprobs = false);

    /**
     * @brief Computes log probabilities for a sequence without generation.
     * @param tokens Input token sequence.
     * @return Log probabilities for each token position.
     * @throws std::runtime_error if computation fails.
     */
    std::vector<float> compute_logprobs(const std::vector<int>& tokens);

    /**
     * @brief Encodes text to token IDs.
     * @param text Input text.
     * @return Vector of token IDs.
     * @throws std::runtime_error if tokenization fails.
     */
    std::vector<int> encode(const std::string& text);

    /**
     * @brief Decodes token IDs to text.
     * @param tokens Vector of token IDs.
     * @return Decoded text.
     * @throws std::runtime_error if detokenization fails.
     */
    std::string decode(const std::vector<int>& tokens);

    /**
     * @brief Resets the internal state (clears KV cache).
     */
    void reset_state();

    /**
     * @brief Gets memory usage information.
     * @return Memory usage in bytes.
     */
    size_t memory_usage() const;

    /**
     * @brief Gets performance statistics.
     * @return Performance statistics as a formatted string.
     */
    std::string performance_stats() const;

private:
    ModelMetadata model_metadata_;          ///< Model metadata
    InferenceConfig config_;                ///< Inference configuration
    std::unique_ptr<core::TensorEngine> tensor_engine_; ///< Tensor computation engine
    std::unique_ptr<class InferenceEngineImpl> impl_;   ///< Implementation details

    /**
     * @brief Initializes the inference engine.
     * @param model_data Loaded model data.
     */
    void initialize(const ModelData& model_data);

    /**
     * @brief Validates input tokens.
     * @param tokens Token sequence to validate.
     * @throws std::runtime_error if tokens are invalid.
     */
    void validate_input_tokens(const std::vector<int>& tokens) const;

    /**
     * @brief Validates batch size.
     * @param batch_size Batch size to validate.
     * @throws std::runtime_error if batch size exceeds limit.
     */
    void validate_batch_size(size_t batch_size) const;

    /**
     * @brief Calculate memory usage of a single tensor.
     * @param tensor The tensor to calculate memory for.
     * @return Memory usage in bytes.
     */
    size_t calculate_tensor_memory(const core::Tensor& tensor) const;

    /**
     * @brief Forward pass through the transformer model.
     * @param tokens Input token sequence.
     * @return Logits tensor for next token prediction.
     */
    core::Tensor forward_pass(const std::vector<int>& tokens);

    /**
     * @brief Sample next token from logits using configured sampling strategy.
     * @param logits Output logits from model.
     * @param logprobs Optional pointer to store log probabilities.
     * @return Sampled token ID.
     */
    int sample_next_token(const core::Tensor& logits, std::vector<float>* logprobs = nullptr);

    /**
     * @brief Apply temperature scaling to logits.
     * @param logits Input logits tensor.
     * @param temperature Temperature value.
     * @return Temperature-scaled logits.
     */
    core::Tensor apply_temperature(const core::Tensor& logits, float temperature);

    /**
     * @brief Apply top-k sampling filter.
     * @param logits Input logits tensor.
     * @param k Top-k value.
     * @return Filtered logits.
     */
    core::Tensor apply_top_k(const core::Tensor& logits, size_t k);

    /**
     * @brief Apply top-p (nucleus) sampling filter.
     * @param logits Input logits tensor.
     * @param p Top-p value.
     * @return Filtered logits.
     */
    core::Tensor apply_top_p(const core::Tensor& logits, float p);

    // Enhanced tokenization helper functions
    
    /**
     * @brief Initialize vocabulary mappings for BPE tokenization.
     */
    void initialize_vocabulary();
    
    /**
     * @brief Split text into words for tokenization.
     * @param text Input text.
     * @return Vector of words.
     */
    std::vector<std::string> split_text_into_words(const std::string& text);
    
    /**
     * @brief Encode a single word using BPE.
     * @param word Input word.
     * @return Vector of token IDs.
     */
    std::vector<int> encode_word_bpe(const std::string& word);
    
    /**
     * @brief Check if a string is punctuation.
     * @param str Input string.
     * @return True if punctuation.
     */
    bool is_punctuation(const std::string& str);
    
    // Vocabulary mappings
    std::unordered_map<std::string, int> vocab_map_;     ///< Token to ID mapping
    std::unordered_map<int, std::string> id_to_token_;   ///< ID to token mapping
    std::vector<std::pair<std::string, std::string>> bpe_merges_; ///< BPE merge rules

    // Beam search helper functions
    
    /**
     * @brief Convert logits to probabilities using softmax.
     * @param logits Input logits vector.
     * @return Probability distribution.
     */
    std::vector<float> softmax(const std::vector<float>& logits);
    
    /**
     * @brief Apply top-k filtering to probabilities.
     * @param probs Input probabilities.
     * @param k Number of top tokens to keep.
     * @return Filtered probabilities.
     */
    std::vector<float> apply_top_k_filtering(const std::vector<float>& probs, size_t k);
    
    /**
     * @brief Apply top-p (nucleus) filtering to probabilities.
     * @param probs Input probabilities.
     * @param p Cumulative probability threshold.
     * @return Filtered probabilities.
     */
    std::vector<float> apply_top_p_filtering(const std::vector<float>& probs, float p);

    /**
     * @brief Beam search helper structure for maintaining candidate sequences.
     */
    struct BeamCandidate {
        std::vector<int> tokens;
        float score;
        float log_prob;
        float normalized_score = 0.0f;  ///< Length-normalized score for ranking
        bool finished;
        
        BeamCandidate(const std::vector<int>& initial_tokens = {})
            : tokens(initial_tokens), score(0.0f), log_prob(0.0f), finished(false) {}
    };

    /**
     * @brief Performs beam search decoding.
     * @param input_tokens Input token sequence.
     * @param max_new_tokens Maximum tokens to generate.
     * @param beam_size Number of beams to maintain.
     * @return Vector of beam candidates sorted by score.
     */
    std::vector<BeamCandidate> beam_search_decode(
        const std::vector<int>& input_tokens,
        size_t max_new_tokens,
        size_t beam_size);
};

/**
 * @brief Convenience function to load a model and create an inference engine.
 * @param model_path Path to the model file.
 * @param config Inference configuration.
 * @return Configured inference engine.
 * @throws std::runtime_error if model loading fails.
 */
std::unique_ptr<InferenceEngine> create_engine(const std::string& model_path,
                                               const InferenceConfig& config = InferenceConfig{});

/**
 * @brief Convenience function for quick text generation.
 * @param model_path Path to the model file.
 * @param prompt Input prompt.
 * @param max_tokens Maximum number of tokens to generate.
 * @param temperature Sampling temperature.
 * @return Generated text.
 * @throws std::runtime_error if generation fails.
 */
std::string quick_generate(const std::string& model_path,
                          const std::string& prompt,
                          size_t max_tokens = 50,
                          float temperature = 1.0f);

} // namespace model
} // namespace turboinfer
