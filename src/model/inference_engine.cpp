/**
 * @file inference_engine.cpp
 * @brief Implementation of the InferenceEngine class with transformer decoder.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/model/inference_engine.hpp"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace turboinfer {
namespace model {

/**
 * @struct KVCache
 * @brief Key-Value cache for transformer attention layers.
 */
struct KVCache {
    std::vector<core::Tensor> key_cache;     ///< Cached key tensors for each layer
    std::vector<core::Tensor> value_cache;   ///< Cached value tensors for each layer
    size_t current_length = 0;               ///< Current sequence length in cache
    size_t max_length = 0;                   ///< Maximum cache length
    
    /**
     * @brief Initialize KV cache for a model.
     * @param num_layers Number of transformer layers.
     * @param max_seq_len Maximum sequence length.
     * @param batch_size Batch size.
     * @param num_heads Number of attention heads.
     * @param head_dim Dimension per attention head.
     */
    void initialize(size_t num_layers, size_t max_seq_len, size_t batch_size, 
                   size_t num_heads, size_t head_dim) {
        max_length = max_seq_len;
        current_length = 0;
        
        key_cache.clear();
        value_cache.clear();
        key_cache.reserve(num_layers);
        value_cache.reserve(num_layers);
        
        for (size_t i = 0; i < num_layers; ++i) {
            // Shape: [batch_size, num_heads, max_seq_len, head_dim]
            core::TensorShape kv_shape({batch_size, num_heads, max_seq_len, head_dim});
            key_cache.emplace_back(kv_shape, core::DataType::kFloat32);
            value_cache.emplace_back(kv_shape, core::DataType::kFloat32);
        }
    }
    
    /**
     * @brief Reset cache for new sequence.
     */
    void reset() {
        current_length = 0;
        // Zero out cache tensors
        for (auto& k : key_cache) {
            std::fill(k.data_ptr<float>(), k.data_ptr<float>() + k.shape().total_size(), 0.0f);
        }
        for (auto& v : value_cache) {
            std::fill(v.data_ptr<float>(), v.data_ptr<float>() + v.shape().total_size(), 0.0f);
        }
    }
    
    /**
     * @brief Update cache with new key-value pairs.
     * @param layer_idx Layer index.
     * @param new_keys New key tensor.
     * @param new_values New value tensor.
     */
    void update(size_t layer_idx, const core::Tensor& new_keys, const core::Tensor& new_values) {
        if (layer_idx >= key_cache.size()) {
            throw std::runtime_error("Layer index out of bounds for KV cache");
        }
        
        // Copy new keys and values into cache at current position
        // This is a simplified implementation - in practice would need proper tensor slicing
        auto& k_cache = key_cache[layer_idx];
        auto& v_cache = value_cache[layer_idx];
        
        // For now, just replace the entire cache (simplified)
        // TODO: Implement proper incremental cache updates
        if (new_keys.shape().total_size() <= k_cache.shape().total_size()) {
            std::copy(new_keys.data_ptr<float>(), 
                     new_keys.data_ptr<float>() + new_keys.shape().total_size(),
                     k_cache.data_ptr<float>());
        }
        
        if (new_values.shape().total_size() <= v_cache.shape().total_size()) {
            std::copy(new_values.data_ptr<float>(), 
                     new_values.data_ptr<float>() + new_values.shape().total_size(),
                     v_cache.data_ptr<float>());
        }
    }
};

/**
 * @struct TransformerLayer
 * @brief Represents a single transformer decoder layer.
 */
struct TransformerLayer {
    std::unique_ptr<core::Tensor> attention_norm;  ///< Pre-attention layer normalization weights
    std::unique_ptr<core::Tensor> q_proj;         ///< Query projection weights
    std::unique_ptr<core::Tensor> k_proj;         ///< Key projection weights  
    std::unique_ptr<core::Tensor> v_proj;         ///< Value projection weights
    std::unique_ptr<core::Tensor> o_proj;         ///< Output projection weights
    std::unique_ptr<core::Tensor> ffn_norm;       ///< Pre-FFN layer normalization weights
    std::unique_ptr<core::Tensor> ffn_up;         ///< FFN up projection weights
    std::unique_ptr<core::Tensor> ffn_down;       ///< FFN down projection weights
    std::unique_ptr<core::Tensor> ffn_gate;       ///< FFN gate projection weights (for SwiGLU)
    
    /**
     * @brief Default constructor.
     */
    TransformerLayer() = default;
    
    /**
     * @brief Forward pass through transformer layer.
     * @param engine Tensor engine for operations.
     * @param input Input tensor.
     * @param kv_cache KV cache for this layer.
     * @param layer_idx Layer index.
     * @param position_ids Position IDs for RoPE.
     * @return Output tensor.
     */
    core::Tensor forward(core::TensorEngine& engine, const core::Tensor& input,
                        KVCache& kv_cache, size_t layer_idx, const core::Tensor& position_ids) {
        // For now, just return the input (placeholder implementation)
        // In a real implementation, this would perform the full transformer layer computation
        
        // 1. Pre-attention normalization
        core::Tensor norm_input = input;
        if (attention_norm) {
            norm_input = engine.rms_norm(input, *attention_norm);
        }
        
        // Simplified: just return input with some basic operations for now
        // This is a placeholder - full transformer layer implementation would be much more complex
        return norm_input;
    }
};

// Forward declaration for implementation
class InferenceEngineImpl {
public:
    ModelData model_data_;                    ///< Loaded model data
    std::vector<TransformerLayer> layers_;    ///< Transformer decoder layers
    std::unique_ptr<core::Tensor> token_embeddings_;  ///< Token embedding weights
    std::unique_ptr<core::Tensor> output_norm_;       ///< Final layer normalization
    std::unique_ptr<core::Tensor> lm_head_;            ///< Language modeling head weights
    KVCache kv_cache_;                        ///< Key-value cache
    std::mt19937 rng_;                        ///< Random number generator for sampling
    
    /**
     * @brief Default constructor.
     */
    InferenceEngineImpl() {
        // Initialize random number generator
        rng_.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    
    /**
     * @brief Initialize the transformer model from loaded data.
     * @param model_data Loaded model data.
     */
    void initialize_model(const ModelData& model_data) {
        model_data_ = model_data;
        
        // Extract model tensors using the public interface
        const core::Tensor* token_embed = model_data.get_tensor("token_embeddings.weight");
        if (!token_embed) {
            token_embed = model_data.get_tensor("embed_tokens.weight");
        }
        if (token_embed) {
            token_embeddings_ = std::make_unique<core::Tensor>(*token_embed);
        }
        
        const core::Tensor* norm = model_data.get_tensor("norm.weight");
        if (!norm) {
            norm = model_data.get_tensor("model.norm.weight");
        }
        if (norm) {
            output_norm_ = std::make_unique<core::Tensor>(*norm);
        }
        
        const core::Tensor* lm_head = model_data.get_tensor("lm_head.weight");
        if (!lm_head) {
            lm_head = model_data.get_tensor("output.weight");
        }
        if (lm_head) {
            lm_head_ = std::make_unique<core::Tensor>(*lm_head);
        }
        
        // TODO: Parse layer-specific weights (e.g., "layers.0.attention.q_proj.weight")
        
        // Initialize layers (placeholder - would need proper weight parsing)
        layers_.resize(model_data.metadata().num_layers);
        
        // Initialize KV cache
        size_t batch_size = 1; // Start with batch size 1
        size_t head_dim = model_data.metadata().hidden_size / model_data.metadata().num_heads;
        kv_cache_.initialize(model_data.metadata().num_layers, 2048, batch_size, 
                           model_data.metadata().num_heads, head_dim);
    }
    
    /**
     * @brief Performs forward pass through the transformer model.
     * @param tokens Input token sequence.
     * @return Logits tensor for next token prediction.
     */
    std::unique_ptr<core::Tensor> forward_pass(const std::vector<int>& tokens) {
        // This is a simplified implementation for demonstration
        // In a real implementation, this would involve:
        // 1. Token embedding lookup
        // 2. Positional encoding addition
        // 3. Forward pass through transformer layers
        // 4. Final layer normalization
        // 5. Language modeling head projection
        
        // For now, return a dummy logits tensor
        size_t vocab_size = 32000; // Common vocab size
        core::TensorShape logits_shape({1, vocab_size});
        auto logits = std::make_unique<core::Tensor>(logits_shape, core::DataType::kFloat32);
        
        // Fill with random values (placeholder)
        auto data_ptr = static_cast<float*>(logits->data());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < vocab_size; ++i) {
            data_ptr[i] = dist(rng_);
        }
        
        return logits;
    }
};

InferenceEngine::InferenceEngine(const ModelData& model_data, const InferenceConfig& config)
    : config_(config), impl_(std::make_unique<InferenceEngineImpl>()) {
    initialize(model_data);
}

InferenceEngine::InferenceEngine(const std::string& model_path, const InferenceConfig& config)
    : config_(config), impl_(std::make_unique<InferenceEngineImpl>()) {
    ModelData model_data = ModelLoader::load(model_path);
    initialize(model_data);
}

InferenceEngine::~InferenceEngine() = default;

InferenceEngine::InferenceEngine(InferenceEngine&& other) noexcept
    : model_metadata_(std::move(other.model_metadata_)),
      config_(std::move(other.config_)),
      tensor_engine_(std::move(other.tensor_engine_)),
      impl_(std::move(other.impl_)) {
}

InferenceEngine& InferenceEngine::operator=(InferenceEngine&& other) noexcept {
    if (this != &other) {
        model_metadata_ = std::move(other.model_metadata_);
        config_ = std::move(other.config_);
        tensor_engine_ = std::move(other.tensor_engine_);
        impl_ = std::move(other.impl_);
    }
    return *this;
}

void InferenceEngine::set_config(const InferenceConfig& config) {
    config_ = config;
}

GenerationResult InferenceEngine::generate(const std::string& prompt, size_t max_new_tokens, bool include_logprobs) {
    std::vector<int> tokens = encode(prompt);
    return generate(tokens, max_new_tokens, include_logprobs);
}

GenerationResult InferenceEngine::generate(const std::vector<int>& input_tokens, size_t max_new_tokens, bool include_logprobs) {
    validate_input_tokens(input_tokens);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    GenerationResult result;
    result.tokens = input_tokens;
    if (include_logprobs) {
        result.logprobs.reserve(input_tokens.size() + max_new_tokens);
    }
    
    // Reset KV cache for new sequence
    impl_->kv_cache_.reset();
    
    // Current sequence
    std::vector<int> current_tokens = input_tokens;
    
    // Generate tokens one by one (autoregressive generation)
    for (size_t i = 0; i < max_new_tokens; ++i) {
        // Forward pass through transformer
        auto logits = forward_pass(current_tokens);
        
        // Apply temperature and sampling
        auto next_token = sample_next_token(logits, include_logprobs ? &result.logprobs : nullptr);
        
        // Add to sequence
        current_tokens.push_back(next_token);
        result.tokens.push_back(next_token);
        
        // Check for end-of-sequence token (assuming token ID 2 is EOS)
        if (next_token == 2) {
            result.finished = true;
            result.stop_reason = "eos_token";
            break;
        }
        
        // Check for maximum sequence length
        if (current_tokens.size() >= config_.max_sequence_length) {
            result.finished = true;
            result.stop_reason = "max_length";
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    result.total_time_ms = static_cast<float>(duration.count());
    size_t generated_tokens = result.tokens.size() - input_tokens.size();
    result.tokens_per_second = generated_tokens / (result.total_time_ms / 1000.0f);
    
    if (!result.finished) {
        result.stop_reason = "max_new_tokens";
    }
    
    return result;
}

std::vector<GenerationResult> InferenceEngine::generate_batch(const std::vector<std::string>& prompts, size_t max_new_tokens, bool include_logprobs) {
    validate_batch_size(prompts.size());
    
    std::vector<GenerationResult> results;
    results.reserve(prompts.size());
    
    for (const auto& prompt : prompts) {
        results.push_back(generate(prompt, max_new_tokens, include_logprobs));
    }
    
    return results;
}

std::vector<GenerationResult> InferenceEngine::generate_batch(const std::vector<std::vector<int>>& input_token_batches, size_t max_new_tokens, bool include_logprobs) {
    validate_batch_size(input_token_batches.size());
    
    std::vector<GenerationResult> results;
    results.reserve(input_token_batches.size());
    
    for (const auto& tokens : input_token_batches) {
        results.push_back(generate(tokens, max_new_tokens, include_logprobs));
    }
    
    return results;
}

std::vector<GenerationResult> InferenceEngine::generate_beam_search(
    const std::vector<int>& input_tokens,
    size_t max_new_tokens,
    size_t beam_size,
    bool include_logprobs) {
    
    if (beam_size == 0) {
        throw std::runtime_error("Beam size must be greater than 0");
    }
    
    validate_input_tokens(input_tokens);
    
    // Perform beam search
    auto beam_candidates = beam_search_decode(input_tokens, max_new_tokens, beam_size);
    
    // Convert beam candidates to generation results
    std::vector<GenerationResult> results;
    results.reserve(beam_candidates.size());
    
    for (const auto& candidate : beam_candidates) {
        GenerationResult result;
        
        // Extract only the newly generated tokens (skip input)
        if (candidate.tokens.size() > input_tokens.size()) {
            result.tokens.assign(
                candidate.tokens.begin() + input_tokens.size(),
                candidate.tokens.end()
            );
        }
        
        result.finished = candidate.finished;
        
        if (include_logprobs) {
            // For beam search, we can provide the average log probability
            result.logprobs.assign(result.tokens.size(), candidate.log_prob / result.tokens.size());
        }
        
        results.push_back(std::move(result));
    }
    
    return results;
}

std::vector<float> InferenceEngine::compute_logprobs(const std::vector<int>& tokens) {
    validate_input_tokens(tokens);
    
    // Placeholder implementation
    std::vector<float> logprobs(tokens.size(), -1.0f);
    return logprobs;
}

std::vector<int> InferenceEngine::encode(const std::string& text) {
    if (vocab_map_.empty()) {
        initialize_vocabulary();
    }
    
    std::vector<int> tokens;
    
    // Split text into words (simplified - in practice you'd handle punctuation better)
    std::vector<std::string> words = split_text_into_words(text);
    
    for (const auto& word : words) {
        auto word_tokens = encode_word_bpe(word);
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
    }
    
    return tokens;
}

std::string InferenceEngine::decode(const std::vector<int>& tokens) {
    if (id_to_token_.empty()) {
        initialize_vocabulary();
    }
    
    std::string text;
    
    for (int token_id : tokens) {
        auto it = id_to_token_.find(token_id);
        if (it != id_to_token_.end()) {
            std::string token_str = it->second;
            
            // Handle BPE merge symbols and special tokens
            if (token_str.find("##") == 0) {
                // Subword continuation (like BERT)
                text += token_str.substr(2);
            } else if (token_str == "<unk>") {
                text += "<?>";
            } else if (token_str == "<pad>") {
                // Skip padding tokens
                continue;
            } else if (token_str == "<s>" || token_str == "</s>") {
                // Skip start/end tokens in detokenization
                continue;
            } else {
                // Regular token - add space before if not the first token and not punctuation
                if (!text.empty() && !is_punctuation(token_str)) {
                    text += " ";
                }
                text += token_str;
            }
        } else {
            // Unknown token ID
            text += "<unk>";
        }
    }
    
    return text;
}

void InferenceEngine::reset_state() {
    // Placeholder implementation
}

size_t InferenceEngine::memory_usage() const {
    // Placeholder implementation
    return 1024 * 1024 * 100; // 100 MB
}

std::string InferenceEngine::performance_stats() const {
    return "Placeholder performance statistics";
}

// Enhanced tokenization helper functions implementation

void InferenceEngine::initialize_vocabulary() {
    // Initialize a basic BPE-style vocabulary
    // In practice, this would be loaded from a vocabulary file (vocab.json + merges.txt)
    
    vocab_map_.clear();
    id_to_token_.clear();
    bpe_merges_.clear();
    
    // Special tokens
    vocab_map_["<unk>"] = 0;
    vocab_map_["<pad>"] = 1;
    vocab_map_["<s>"] = 2;
    vocab_map_["</s>"] = 3;
    
    id_to_token_[0] = "<unk>";
    id_to_token_[1] = "<pad>";
    id_to_token_[2] = "<s>";
    id_to_token_[3] = "</s>";
    
    int token_id = 4;
    
    // Basic character-level tokens (for fallback)
    for (int i = 0; i < 256; ++i) {
        std::string char_token = std::string(1, static_cast<char>(i));
        vocab_map_[char_token] = token_id;
        id_to_token_[token_id] = char_token;
        token_id++;
    }
    
    // Common English subwords and tokens (simplified BPE vocab)
    std::vector<std::string> common_tokens = {
        // Common subwords
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one",
        "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see",
        "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
        
        // Common prefixes and suffixes
        "ing", "ed", "er", "ly", "un", "re", "in", "dis", "en", "non", "over", "pre", "under",
        "##ing", "##ed", "##er", "##ly", "##s", "##d", "##t", "##n", "##r", "##l",
        
        // Common punctuation as tokens
        ".", ",", "!", "?", ";", ":", "'", "\"", "(", ")", "-", "_", "/", "\\",
        
        // Whitespace
        " ", "\n", "\t",
        
        // Common word pieces
        "an", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it", "me", "my", "no", "of",
        "on", "or", "so", "to", "up", "we", "as", "am", "us", "ex", "oh", "ok", "hi",
        
        // More complex subwords
        "tion", "able", "ment", "ness", "less", "ful", "ant", "ent", "ive", "ous", "ish", "est"
    };
    
    for (const auto& token : common_tokens) {
        vocab_map_[token] = token_id;
        id_to_token_[token_id] = token;
        token_id++;
    }
    
    // Add some BPE merge rules (simplified)
    bpe_merges_ = {
        {"t", "h"},     // th
        {"i", "n"},     // in  
        {"a", "n"},     // an
        {"e", "r"},     // er
        {"o", "n"},     // on
        {"r", "e"},     // re
        {"th", "e"},    // the
        {"in", "g"},    // ing
        {"a", "nd"},    // and
        {"t", "o"},     // to
    };
}

std::vector<std::string> InferenceEngine::split_text_into_words(const std::string& text) {
    std::vector<std::string> words;
    std::string current_word;
    
    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        
        if (std::isspace(c)) {
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
            // Add whitespace as a separate token
            words.push_back(std::string(1, c));
        } else if (std::ispunct(c)) {
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
            // Add punctuation as a separate token
            words.push_back(std::string(1, c));
        } else {
            current_word += c;
        }
    }
    
    if (!current_word.empty()) {
        words.push_back(current_word);
    }
    
    return words;
}

std::vector<int> InferenceEngine::encode_word_bpe(const std::string& word) {
    if (word.empty()) {
        return {};
    }
    
    // Check if the whole word is in vocabulary
    auto it = vocab_map_.find(word);
    if (it != vocab_map_.end()) {
        return {it->second};
    }
    
    // Try to apply BPE merges
    std::vector<std::string> word_tokens;
    
    // Start with character-level tokenization
    for (char c : word) {
        word_tokens.push_back(std::string(1, c));
    }
    
    // Apply BPE merges
    bool changed = true;
    while (changed && word_tokens.size() > 1) {
        changed = false;
        
        for (const auto& merge : bpe_merges_) {
            for (size_t i = 0; i < word_tokens.size() - 1; ++i) {
                if (word_tokens[i] == merge.first && word_tokens[i + 1] == merge.second) {
                    std::string merged = merge.first + merge.second;
                    word_tokens[i] = merged;
                    word_tokens.erase(word_tokens.begin() + i + 1);
                    changed = true;
                    break;
                }
            }
            if (changed) break;
        }
    }
    
    // Convert tokens to IDs
    std::vector<int> token_ids;
    for (const auto& token : word_tokens) {
        auto token_it = vocab_map_.find(token);
        if (token_it != vocab_map_.end()) {
            token_ids.push_back(token_it->second);
        } else {
            // Fall back to character-level if token not found
            for (char c : token) {
                std::string char_token(1, c);
                auto char_it = vocab_map_.find(char_token);
                if (char_it != vocab_map_.end()) {
                    token_ids.push_back(char_it->second);
                } else {
                    token_ids.push_back(0); // <unk>
                }
            }
        }
    }
    
    return token_ids;
}

bool InferenceEngine::is_punctuation(const std::string& str) {
    if (str.empty()) return false;
    
    // Common punctuation characters
    static const std::string punct = ".,!?;:\"'()[]{}/-_\\";
    return str.length() == 1 && punct.find(str[0]) != std::string::npos;
}

void InferenceEngine::initialize(const ModelData& model_data) {
    model_metadata_ = model_data.metadata();
    tensor_engine_ = std::make_unique<core::TensorEngine>(config_.device);
    
    // Initialize the transformer model
    impl_->initialize_model(model_data);
}

void InferenceEngine::validate_input_tokens(const std::vector<int>& tokens) const {
    if (tokens.empty()) {
        throw std::runtime_error("Input tokens cannot be empty");
    }
    
    if (tokens.size() > config_.max_sequence_length) {
        throw std::runtime_error("Input sequence length exceeds maximum allowed length");
    }
}

void InferenceEngine::validate_batch_size(size_t batch_size) const {
    if (batch_size == 0) {
        throw std::runtime_error("Batch size cannot be zero");
    }
    
    if (batch_size > config_.max_batch_size) {
        throw std::runtime_error("Batch size exceeds maximum allowed batch size");
    }
}

core::Tensor InferenceEngine::forward_pass(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        throw std::runtime_error("Cannot perform forward pass with empty token sequence");
    }
    
    // 1. Token embedding lookup
    core::TensorShape input_shape({1, tokens.size()}); // Batch size = 1
    core::Tensor input_ids(input_shape, core::DataType::kInt32);
    std::copy(tokens.begin(), tokens.end(), input_ids.data_ptr<int>());
    
    // For simplicity, create a dummy embedding tensor
    // In practice, this would be an embedding lookup
    core::TensorShape embed_shape({1, tokens.size(), static_cast<size_t>(model_metadata_.hidden_size)});
    core::Tensor embeddings(embed_shape, core::DataType::kFloat32);
    
    // Initialize embeddings with simple values (placeholder)
    float* embed_data = embeddings.data_ptr<float>();
    for (size_t i = 0; i < embeddings.shape().total_size(); ++i) {
        embed_data[i] = 0.1f * (i % 100); // Simple initialization
    }
    
    // 2. Create position IDs for RoPE
    core::TensorShape pos_shape({tokens.size()});
    core::Tensor position_ids(pos_shape, core::DataType::kFloat32);
    float* pos_data = position_ids.data_ptr<float>();
    for (size_t i = 0; i < tokens.size(); ++i) {
        pos_data[i] = static_cast<float>(i);
    }
    
    // 3. Forward pass through transformer layers
    core::Tensor hidden_states = embeddings;
    
    for (size_t layer_idx = 0; layer_idx < impl_->layers_.size(); ++layer_idx) {
        if (layer_idx < impl_->layers_.size()) {
            hidden_states = impl_->layers_[layer_idx].forward(
                *tensor_engine_, hidden_states, impl_->kv_cache_, layer_idx, position_ids);
        }
    }
    
    // 4. Final layer normalization
    if (impl_->output_norm_) {
        hidden_states = tensor_engine_->rms_norm(hidden_states, *impl_->output_norm_);
    }
    
    // 5. Language modeling head (project to vocabulary)
    core::TensorShape logits_shape({1, tokens.size(), static_cast<size_t>(model_metadata_.vocab_size)});
    core::Tensor logits(logits_shape, core::DataType::kFloat32);
    
    if (impl_->lm_head_) {
        logits = tensor_engine_->matmul(hidden_states, *impl_->lm_head_);
    } else {
        // Fallback: create dummy logits
        float* logits_data = logits.data_ptr<float>();
        for (size_t i = 0; i < logits.shape().total_size(); ++i) {
            logits_data[i] = static_cast<float>(impl_->rng_()) / impl_->rng_.max() - 0.5f;
        }
    }
    
    // 6. Extract logits for the last token (for next token prediction)
    // For simplicity, we'll return the full logits tensor
    // In practice, we'd slice to get only the last position
    return logits;
}

int InferenceEngine::sample_next_token(const core::Tensor& logits, std::vector<float>* logprobs) {
    if (logits.empty()) {
        throw std::runtime_error("Cannot sample from empty logits");
    }
    
    // For simplicity, assume logits shape is [1, seq_len, vocab_size]
    // Extract logits for the last position
    auto logits_shape = logits.shape();
    if (logits_shape.ndim() < 2) {
        throw std::runtime_error("Logits tensor must have at least 2 dimensions");
    }
    
    size_t vocab_size = logits_shape.size(logits_shape.ndim() - 1);
    size_t seq_len = logits_shape.ndim() >= 3 ? logits_shape.size(logits_shape.ndim() - 2) : 1;
    
    // Get logits for the last token position
    const float* logits_data = logits.data_ptr<float>();
    size_t last_token_offset = (seq_len - 1) * vocab_size;
    
    // Copy last token logits
    std::vector<float> token_logits(vocab_size);
    std::copy(logits_data + last_token_offset, logits_data + last_token_offset + vocab_size, token_logits.begin());
    
    // Apply temperature
    if (config_.temperature != 1.0f && config_.temperature > 0.0f) {
        for (float& logit : token_logits) {
            logit /= config_.temperature;
        }
    }
    
    // Apply top-k filtering
    if (config_.top_k > 0 && config_.top_k < vocab_size) {
        // Sort indices by logit values
        std::vector<std::pair<float, int>> logit_pairs;
        for (size_t i = 0; i < vocab_size; ++i) {
            logit_pairs.emplace_back(token_logits[i], static_cast<int>(i));
        }
        std::sort(logit_pairs.begin(), logit_pairs.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Zero out logits beyond top-k
        for (size_t i = config_.top_k; i < vocab_size; ++i) {
            token_logits[logit_pairs[i].second] = -std::numeric_limits<float>::infinity();
        }
    }
    
    // Convert logits to probabilities using softmax
    float max_logit = *std::max_element(token_logits.begin(), token_logits.end());
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(token_logits[i] - max_logit);
        sum += probs[i];
    }
    
    for (float& prob : probs) {
        prob /= sum;
    }
    
    // Apply top-p filtering
    if (config_.top_p < 1.0f) {
        // Sort probabilities
        std::vector<std::pair<float, int>> prob_pairs;
        for (size_t i = 0; i < vocab_size; ++i) {
            prob_pairs.emplace_back(probs[i], static_cast<int>(i));
        }
        std::sort(prob_pairs.begin(), prob_pairs.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Find cutoff point
        float cumsum = 0.0f;
        size_t cutoff = vocab_size;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumsum += prob_pairs[i].first;
            if (cumsum >= config_.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Zero out probabilities beyond cutoff
        for (size_t i = cutoff; i < vocab_size; ++i) {
            probs[prob_pairs[i].second] = 0.0f;
        }
        
        // Renormalize
        float new_sum = 0.0f;
        for (float prob : probs) {
            new_sum += prob;
        }
        if (new_sum > 0.0f) {
            for (float& prob : probs) {
                prob /= new_sum;
            }
        }
    }
    
    // Sample from the probability distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float random_value = dist(impl_->rng_);
    
    float cumsum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (random_value <= cumsum) {
            // Store log probability if requested
            if (logprobs) {
                logprobs->push_back(std::log(probs[i]));
            }
            return static_cast<int>(i);
        }
    }
    
    // Fallback: return last token
    if (logprobs) {
        logprobs->push_back(std::log(probs[vocab_size - 1]));
    }
    return static_cast<int>(vocab_size - 1);
}

core::Tensor InferenceEngine::apply_temperature(const core::Tensor& logits, float temperature) {
    if (temperature <= 0.0f) {
        throw std::runtime_error("Temperature must be positive");
    }
    
    if (temperature == 1.0f) {
        return logits; // No scaling needed
    }
    
    // Create a copy and scale by temperature
    core::Tensor scaled_logits(logits.shape(), logits.dtype());
    const float* input_data = logits.data_ptr<float>();
    float* output_data = scaled_logits.data_ptr<float>();
    
    size_t total_elements = logits.shape().total_size();
    for (size_t i = 0; i < total_elements; ++i) {
        output_data[i] = input_data[i] / temperature;
    }
    
    return scaled_logits;
}

core::Tensor InferenceEngine::apply_top_k(const core::Tensor& logits, size_t k) {
    // This is a simplified implementation
    // In practice, would need more sophisticated tensor operations
    return logits; // Placeholder
}

core::Tensor InferenceEngine::apply_top_p(const core::Tensor& logits, float p) {
    // This is a simplified implementation
    // In practice, would need more sophisticated tensor operations  
    return logits; // Placeholder
}

std::vector<InferenceEngine::BeamCandidate> InferenceEngine::beam_search_decode(
    const std::vector<int>& input_tokens,
    size_t max_new_tokens,
    size_t beam_size) {
    
    // Initialize beams with the input sequence
    std::vector<BeamCandidate> beams;
    beams.emplace_back(input_tokens);
    
    // Track finished beams separately
    std::vector<BeamCandidate> finished_beams;
    
    for (size_t step = 0; step < max_new_tokens; ++step) {
        std::vector<BeamCandidate> new_beams;
        
        for (auto& beam : beams) {
            if (beam.finished) {
                finished_beams.push_back(beam);
                continue;
            }
            
            // Get logits for current beam
            auto logits = impl_->forward_pass(beam.tokens);
            
            // Apply temperature if configured
            if (config_.temperature != 1.0f) {
                auto temp_logits = apply_temperature(*logits, config_.temperature);
                logits = std::make_unique<core::Tensor>(std::move(temp_logits));
            }
            
            // Get top beam_size tokens
            auto data_ptr = static_cast<const float*>(logits->data());
            size_t vocab_size = logits->shape().dimensions().back();
            
            // Create token-probability pairs
            std::vector<std::pair<int, float>> token_probs;
            token_probs.reserve(vocab_size);
            
            for (size_t i = 0; i < vocab_size; ++i) {
                token_probs.emplace_back(static_cast<int>(i), data_ptr[i]);
            }
            
            // Sort by probability (descending)
            std::sort(token_probs.begin(), token_probs.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            // Take top beam_size candidates
            size_t candidates_to_consider = std::min(beam_size, token_probs.size());
            for (size_t i = 0; i < candidates_to_consider; ++i) {
                BeamCandidate new_beam = beam;
                new_beam.tokens.push_back(token_probs[i].first);
                new_beam.log_prob += std::log(std::max(token_probs[i].second, 1e-10f));
                new_beam.score = new_beam.log_prob / new_beam.tokens.size(); // Length normalized
                
                // Check if sequence is finished (EOS token)
                if (token_probs[i].first == 2) { // Assuming EOS token is 2
                    new_beam.finished = true;
                }
                
                new_beams.push_back(new_beam);
            }
        }
        
        // If no active beams, break
        if (new_beams.empty()) {
            break;
        }
        
        // Sort all new beams by score and keep top beam_size
        std::sort(new_beams.begin(), new_beams.end(),
                 [](const BeamCandidate& a, const BeamCandidate& b) {
                     return a.score > b.score;
                 });
        
        // Keep only top beam_size beams
        if (new_beams.size() > beam_size) {
            new_beams.resize(beam_size);
        }
        
        beams = std::move(new_beams);
        
        // Early stopping if all beams are finished
        bool all_finished = std::all_of(beams.begin(), beams.end(),
                                       [](const BeamCandidate& b) { return b.finished; });
        if (all_finished) {
            break;
        }
    }
    
    // Combine finished and unfinished beams
    finished_beams.insert(finished_beams.end(), beams.begin(), beams.end());
    
    // Sort by score and return
    std::sort(finished_beams.begin(), finished_beams.end(),
             [](const BeamCandidate& a, const BeamCandidate& b) {
                 return a.score > b.score;
             });
    
    return finished_beams;
}

// Convenience functions

std::unique_ptr<InferenceEngine> create_engine(const std::string& model_path, const InferenceConfig& config) {
    return std::make_unique<InferenceEngine>(model_path, config);
}

std::string quick_generate(const std::string& model_path, const std::string& prompt, size_t max_tokens, float temperature) {
    InferenceConfig config;
    config.temperature = temperature;
    
    InferenceEngine engine(model_path, config);
    auto result = engine.generate(prompt, max_tokens);
    return engine.decode(result.tokens);
}

} // namespace model
} // namespace turboinfer
