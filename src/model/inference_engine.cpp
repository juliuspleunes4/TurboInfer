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
#include <queue>
#include <cmath>
#include <sstream>
#include <iomanip>

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
     * @brief Update cache with new key-value pairs using incremental updates.
     * @param layer_idx Layer index.
     * @param new_keys New key tensor for current token(s).
     * @param new_values New value tensor for current token(s).
     * @return Pair of (full_keys, full_values) including cached and new tokens.
     */
    std::pair<core::Tensor, core::Tensor> update_incremental(size_t layer_idx, 
                                                             const core::Tensor& new_keys, 
                                                             const core::Tensor& new_values) {
        if (layer_idx >= key_cache.size()) {
            throw std::runtime_error("Layer index out of bounds for KV cache");
        }
        
        auto& k_cache = key_cache[layer_idx];
        auto& v_cache = value_cache[layer_idx];
        
        // Get cache dimensions: [batch_size, num_heads, max_seq_len, head_dim]
        const auto& cache_shape = k_cache.shape();
        size_t batch_size = cache_shape.size(0);
        size_t num_heads = cache_shape.size(1);
        size_t max_seq_len = cache_shape.size(2);
        size_t head_dim = cache_shape.size(3);
        
        // Get new token dimensions: [batch_size, num_heads, new_tokens, head_dim]
        const auto& new_shape = new_keys.shape();
        size_t new_tokens = new_shape.size(2);
        
        // Check if we have space in cache
        if (current_length + new_tokens > max_seq_len) {
            throw std::runtime_error("KV cache overflow: sequence too long");
        }
        
        // Copy new keys/values into cache at current position
        float* k_cache_ptr = k_cache.data_ptr<float>();
        float* v_cache_ptr = v_cache.data_ptr<float>();
        const float* new_k_ptr = new_keys.data_ptr<float>();
        const float* new_v_ptr = new_values.data_ptr<float>();
        
        // Calculate offset for insertion position
        size_t offset = current_length * head_dim;
        size_t layer_stride = num_heads * max_seq_len * head_dim;
        size_t head_stride = max_seq_len * head_dim;
        size_t new_layer_stride = num_heads * new_tokens * head_dim;
        size_t new_head_stride = new_tokens * head_dim;
        
        // Copy for each batch and head
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                // Calculate pointers for this batch/head
                float* k_dst = k_cache_ptr + b * layer_stride + h * head_stride + offset;
                float* v_dst = v_cache_ptr + b * layer_stride + h * head_stride + offset;
                const float* k_src = new_k_ptr + b * new_layer_stride + h * new_head_stride;
                const float* v_src = new_v_ptr + b * new_layer_stride + h * new_head_stride;
                
                // Copy new tokens into cache
                std::copy(k_src, k_src + new_tokens * head_dim, k_dst);
                std::copy(v_src, v_src + new_tokens * head_dim, v_dst);
            }
        }
        
        // Update current length
        current_length += new_tokens;
        
        // Return sliced cache up to current length for attention computation
        // Shape: [batch_size, num_heads, current_length, head_dim]
        core::TensorShape result_shape({batch_size, num_heads, current_length, head_dim});
        core::Tensor full_keys(result_shape, core::DataType::kFloat32);
        core::Tensor full_values(result_shape, core::DataType::kFloat32);
        
        // Copy current cache contents
        float* result_k_ptr = full_keys.data_ptr<float>();
        float* result_v_ptr = full_values.data_ptr<float>();
        size_t result_layer_stride = num_heads * current_length * head_dim;
        size_t result_head_stride = current_length * head_dim;
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                const float* cache_k_src = k_cache_ptr + b * layer_stride + h * head_stride;
                const float* cache_v_src = v_cache_ptr + b * layer_stride + h * head_stride;
                float* result_k_dst = result_k_ptr + b * result_layer_stride + h * result_head_stride;
                float* result_v_dst = result_v_ptr + b * result_layer_stride + h * result_head_stride;
                
                std::copy(cache_k_src, cache_k_src + current_length * head_dim, result_k_dst);
                std::copy(cache_v_src, cache_v_src + current_length * head_dim, result_v_dst);
            }
        }
        
        return {std::move(full_keys), std::move(full_values)};
    }
    
    /**
     * @brief Legacy update method (maintained for compatibility).
     * @param layer_idx Layer index.
     * @param new_keys New key tensor.
     * @param new_values New value tensor.
     */
    void update(size_t layer_idx, const core::Tensor& new_keys, const core::Tensor& new_values) {
        // Use the new incremental method
        update_incremental(layer_idx, new_keys, new_values);
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
        // Store input for residual connection
        core::Tensor residual = input;
        
        // 1. Pre-attention normalization (RMSNorm)
        core::Tensor norm_input = input;
        if (attention_norm) {
            norm_input = engine.rms_norm(input, *attention_norm);
        }
        
        // 2. Multi-head self-attention
        core::Tensor attn_output = compute_attention(engine, norm_input, kv_cache, layer_idx, position_ids);
        
        // 3. First residual connection
        core::Tensor post_attn = engine.add(residual, attn_output);
        
        // 4. Pre-FFN normalization (RMSNorm)
        core::Tensor ffn_input = post_attn;
        if (ffn_norm) {
            ffn_input = engine.rms_norm(post_attn, *ffn_norm);
        }
        
        // 5. Feed-forward network with SwiGLU activation
        core::Tensor ffn_output = compute_ffn(engine, ffn_input);
        
        // 6. Second residual connection
        core::Tensor output = engine.add(post_attn, ffn_output);
        
        return output;
    }
    
private:
    /**
     * @brief Compute multi-head self-attention.
     * @param engine Tensor engine for operations.
     * @param input Input tensor [seq_len, hidden_size].
     * @param kv_cache KV cache for this layer.
     * @param layer_idx Layer index.
     * @param position_ids Position IDs for RoPE.
     * @return Attention output tensor.
     */
    core::Tensor compute_attention(core::TensorEngine& engine, const core::Tensor& input,
                                 KVCache& kv_cache, size_t layer_idx, const core::Tensor& position_ids) {
        if (!q_proj || !k_proj || !v_proj || !o_proj) {
            // If attention weights are not available, return input (fallback)
            return input;
        }
        
        // Project to Q, K, V using matmul (linear transformation)
        core::Tensor q = engine.matmul(input, *q_proj);  // [seq_len, hidden_size]
        core::Tensor k = engine.matmul(input, *k_proj);
        core::Tensor v = engine.matmul(input, *v_proj);
        
        // Reshape for multi-head attention
        // For simplicity, assume single head for now (can be extended)
        size_t seq_len = input.shape().size(0);
        size_t hidden_size = input.shape().size(1);
        
        // Apply RoPE (Rotary Position Embedding) if position_ids provided
        if (position_ids.shape().total_size() > 0) {
            q = apply_rope(engine, q, position_ids);
            k = apply_rope(engine, k, position_ids);
        }
        
        // Update KV cache incrementally and get full cached keys/values
        if (layer_idx < kv_cache.key_cache.size()) {
            auto cached_kv = kv_cache.update_incremental(layer_idx, k, v);
            // Use cached K,V which includes previous tokens + current token
            k = std::move(cached_kv.first);
            v = std::move(cached_kv.second);
        }
        
        // Compute attention scores: Q @ K^T
        core::Tensor k_transposed = engine.transpose(k);
        core::Tensor scores = engine.matmul(q, k_transposed);
        
        // Scale by sqrt(d_k)
        float scale_factor = 1.0f / std::sqrt(static_cast<float>(hidden_size));
        core::Tensor scaled_scores = engine.scale(scores, scale_factor);
        
        // Apply causal mask (for autoregressive generation)
        core::Tensor masked_scores = apply_causal_mask(engine, scaled_scores);
        
        // Softmax
        core::Tensor attn_weights = engine.softmax(masked_scores);
        
        // Apply attention to values: Attention @ V
        core::Tensor attn_output = engine.matmul(attn_weights, v);
        
        // Output projection
        core::Tensor output = engine.matmul(attn_output, *o_proj);
        
        return output;
    }
    
    /**
     * @brief Compute feed-forward network with SwiGLU activation.
     * @param engine Tensor engine for operations.
     * @param input Input tensor.
     * @return FFN output tensor.
     */
    core::Tensor compute_ffn(core::TensorEngine& engine, const core::Tensor& input) {
        if (!ffn_up || !ffn_down) {
            // If FFN weights are not available, return input (fallback)
            return input;
        }
        
        // Up projection (linear transformation)
        core::Tensor up_output = engine.matmul(input, *ffn_up);
        
        // SwiGLU activation: x * swish(gate(x))
        core::Tensor activated = up_output; // Initialize with up_output
        if (ffn_gate) {
            // Full SwiGLU with separate gate projection
            core::Tensor gate_output = engine.matmul(input, *ffn_gate);
            core::Tensor swish_gate = engine.silu(gate_output); // Use silu instead of swish
            activated = engine.multiply(up_output, swish_gate);
        } else {
            // Simplified: just apply ReLU activation
            activated = engine.relu(up_output);
        }
        
        // Down projection
        core::Tensor output = engine.matmul(activated, *ffn_down);
        
        return output;
    }
    
    /**
     * @brief Apply RoPE (Rotary Position Embedding).
     * @param engine Tensor engine.
     * @param input Input tensor.
     * @param position_ids Position IDs.
     * @return Tensor with RoPE applied.
     */
    core::Tensor apply_rope(core::TensorEngine& engine, const core::Tensor& input, const core::Tensor& position_ids) {
        // Simplified RoPE implementation
        // In practice, this would involve complex trigonometric operations
        // For now, return input unchanged (can be enhanced later)
        (void)engine;
        (void)position_ids;
        return input;
    }
    
    /**
     * @brief Apply causal mask to attention scores.
     * @param engine Tensor engine.
     * @param scores Attention scores.
     * @return Masked scores.
     */
    core::Tensor apply_causal_mask(core::TensorEngine& engine, const core::Tensor& scores) {
        // Apply causal mask to prevent attending to future tokens
        size_t seq_len = scores.shape().size(0);
        
        // Create a copy of scores for masking
        core::Tensor masked_scores = scores;
        float* data = masked_scores.data_ptr<float>();
        
        // Apply mask: set upper triangular part to -inf
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = i + 1; j < seq_len; ++j) {
                size_t idx = i * seq_len + j;
                data[idx] = -std::numeric_limits<float>::infinity();
            }
        }
        
        return masked_scores;
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
    
    // Performance tracking
    mutable size_t total_generations_ = 0;    ///< Total number of generations performed
    mutable size_t total_tokens_generated_ = 0; ///< Total tokens generated
    mutable float total_generation_time_ms_ = 0.0f; ///< Total time spent in generation
    mutable float total_forward_time_ms_ = 0.0f;    ///< Total time spent in forward passes
    mutable size_t total_forward_passes_ = 0;       ///< Total number of forward passes
    mutable float average_tokens_per_second_ = 0.0f; ///< Running average of tokens/second
    mutable float peak_tokens_per_second_ = 0.0f;    ///< Peak tokens/second achieved
    mutable size_t cache_hits_ = 0;                  ///< KV-cache efficiency hits
    mutable size_t cache_misses_ = 0;                ///< KV-cache efficiency misses
    
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
        
        // Parse layer-specific weights for transformer layers
        layers_.resize(model_data.metadata().num_layers);
        
        for (size_t layer_idx = 0; layer_idx < model_data.metadata().num_layers; ++layer_idx) {
            auto& layer = layers_[layer_idx];
            
            // Load attention weights
            std::string layer_prefix = "model.layers." + std::to_string(layer_idx) + ".";
            std::string alt_prefix = "layers." + std::to_string(layer_idx) + ".";
            
            // Try different naming conventions for attention weights
            const core::Tensor* q_proj = model_data.get_tensor(layer_prefix + "self_attn.q_proj.weight");
            if (!q_proj) q_proj = model_data.get_tensor(alt_prefix + "attention.q_proj.weight");
            if (!q_proj) q_proj = model_data.get_tensor(alt_prefix + "self_attn.q_proj.weight");
            if (q_proj) layer.q_proj = std::make_unique<core::Tensor>(*q_proj);
            
            const core::Tensor* k_proj = model_data.get_tensor(layer_prefix + "self_attn.k_proj.weight");
            if (!k_proj) k_proj = model_data.get_tensor(alt_prefix + "attention.k_proj.weight");
            if (!k_proj) k_proj = model_data.get_tensor(alt_prefix + "self_attn.k_proj.weight");
            if (k_proj) layer.k_proj = std::make_unique<core::Tensor>(*k_proj);
            
            const core::Tensor* v_proj = model_data.get_tensor(layer_prefix + "self_attn.v_proj.weight");
            if (!v_proj) v_proj = model_data.get_tensor(alt_prefix + "attention.v_proj.weight");
            if (!v_proj) v_proj = model_data.get_tensor(alt_prefix + "self_attn.v_proj.weight");
            if (v_proj) layer.v_proj = std::make_unique<core::Tensor>(*v_proj);
            
            const core::Tensor* o_proj = model_data.get_tensor(layer_prefix + "self_attn.o_proj.weight");
            if (!o_proj) o_proj = model_data.get_tensor(alt_prefix + "attention.o_proj.weight");
            if (!o_proj) o_proj = model_data.get_tensor(alt_prefix + "self_attn.o_proj.weight");
            if (o_proj) layer.o_proj = std::make_unique<core::Tensor>(*o_proj);
            
            // Load normalization weights
            const core::Tensor* attn_norm = model_data.get_tensor(layer_prefix + "input_layernorm.weight");
            if (!attn_norm) attn_norm = model_data.get_tensor(alt_prefix + "attention_norm.weight");
            if (!attn_norm) attn_norm = model_data.get_tensor(alt_prefix + "input_layernorm.weight");
            if (attn_norm) layer.attention_norm = std::make_unique<core::Tensor>(*attn_norm);
            
            const core::Tensor* ffn_norm = model_data.get_tensor(layer_prefix + "post_attention_layernorm.weight");
            if (!ffn_norm) ffn_norm = model_data.get_tensor(alt_prefix + "ffn_norm.weight");
            if (!ffn_norm) ffn_norm = model_data.get_tensor(alt_prefix + "post_attention_layernorm.weight");
            if (ffn_norm) layer.ffn_norm = std::make_unique<core::Tensor>(*ffn_norm);
            
            // Load FFN weights (try different naming conventions)
            const core::Tensor* ffn_up = model_data.get_tensor(layer_prefix + "mlp.up_proj.weight");
            if (!ffn_up) ffn_up = model_data.get_tensor(alt_prefix + "feed_forward.w1.weight");
            if (!ffn_up) ffn_up = model_data.get_tensor(alt_prefix + "mlp.up_proj.weight");
            if (ffn_up) layer.ffn_up = std::make_unique<core::Tensor>(*ffn_up);
            
            const core::Tensor* ffn_down = model_data.get_tensor(layer_prefix + "mlp.down_proj.weight");
            if (!ffn_down) ffn_down = model_data.get_tensor(alt_prefix + "feed_forward.w2.weight");
            if (!ffn_down) ffn_down = model_data.get_tensor(alt_prefix + "mlp.down_proj.weight");
            if (ffn_down) layer.ffn_down = std::make_unique<core::Tensor>(*ffn_down);
            
            const core::Tensor* ffn_gate = model_data.get_tensor(layer_prefix + "mlp.gate_proj.weight");
            if (!ffn_gate) ffn_gate = model_data.get_tensor(alt_prefix + "feed_forward.w3.weight");
            if (!ffn_gate) ffn_gate = model_data.get_tensor(alt_prefix + "mlp.gate_proj.weight");
            if (ffn_gate) layer.ffn_gate = std::make_unique<core::Tensor>(*ffn_gate);
        }
        
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
        auto forward_start = std::chrono::high_resolution_clock::now();
        
        // Track forward pass count
        total_forward_passes_++;
        
        if (tokens.empty()) {
            throw std::runtime_error("Cannot perform forward pass with empty token sequence");
        }
        
        // 1. Token embedding lookup (simplified)
        size_t hidden_size = model_data_.metadata().hidden_size;
        size_t seq_len = tokens.size();
        core::TensorShape embed_shape({seq_len, hidden_size});
        core::Tensor embeddings(embed_shape, core::DataType::kFloat32);
        
        if (token_embeddings_) {
            // Real embedding lookup from learned weights
            float* embed_data = embeddings.data_ptr<float>();
            const float* weight_data = token_embeddings_->data_ptr<float>();
            size_t vocab_size = token_embeddings_->shape().size(0);
            
            for (size_t i = 0; i < seq_len; ++i) {
                int token_id = tokens[i];
                if (token_id >= 0 && static_cast<size_t>(token_id) < vocab_size) {
                    // Copy embedding vector for this token
                    const float* token_embed = weight_data + token_id * hidden_size;
                    float* output_embed = embed_data + i * hidden_size;
                    std::copy(token_embed, token_embed + hidden_size, output_embed);
                } else {
                    // Handle out-of-vocabulary tokens with zero embeddings
                    float* output_embed = embed_data + i * hidden_size;
                    std::fill(output_embed, output_embed + hidden_size, 0.0f);
                }
            }
        } else {
            // Fallback: create simple embeddings based on token IDs
            float* embed_data = embeddings.data_ptr<float>();
            for (size_t i = 0; i < seq_len; ++i) {
                int token_id = tokens[i];
                for (size_t h = 0; h < hidden_size; ++h) {
                    embed_data[i * hidden_size + h] = 0.1f * std::sin(static_cast<float>(token_id + h) * 0.01f);
                }
            }
        }
        
        // 2. Simple transformer processing (placeholder for now)
        core::Tensor hidden_states = embeddings;
        
        // 3. Language modeling head projection
        size_t vocab_size = model_data_.metadata().vocab_size > 0 ? model_data_.metadata().vocab_size : 32000;
        core::TensorShape logits_shape({1, vocab_size});
        core::Tensor logits(logits_shape, core::DataType::kFloat32);
        float* logits_data = logits.data_ptr<float>();
        
        if (lm_head_) {
            // Real LM head projection (simplified)
            const float* hidden_data = hidden_states.data_ptr<float>();
            const float* weight_data = lm_head_->data_ptr<float>();
            
            // Take last token's hidden state
            size_t last_token_offset = (seq_len - 1) * hidden_size;
            
            // Simple matrix multiplication: logits = hidden @ lm_head.T
            for (size_t v = 0; v < vocab_size && v < lm_head_->shape().size(1); ++v) {
                float logit_val = 0.0f;
                for (size_t h = 0; h < hidden_size && h < lm_head_->shape().size(0); ++h) {
                    logit_val += hidden_data[last_token_offset + h] * weight_data[h * vocab_size + v];
                }
                logits_data[v] = logit_val;
            }
        } else {
            // Fallback: create logits with learned patterns
            const float* hidden_data = hidden_states.data_ptr<float>();
            size_t last_token_offset = (seq_len - 1) * hidden_size;
            
            for (size_t v = 0; v < vocab_size; ++v) {
                float logit_val = 0.0f;
                for (size_t h = 0; h < std::min(hidden_size, size_t(512)); ++h) {
                    if (last_token_offset + h < hidden_states.shape().total_size()) {
                        logit_val += hidden_data[last_token_offset + h] * std::sin(static_cast<float>(v + h) * 0.01f);
                    }
                }
                logits_data[v] = logit_val;
            }
        }
        
        auto forward_end = std::chrono::high_resolution_clock::now();
        auto forward_duration = std::chrono::duration_cast<std::chrono::milliseconds>(forward_end - forward_start);
        total_forward_time_ms_ += static_cast<float>(forward_duration.count());
        
        return std::make_unique<core::Tensor>(std::move(logits));
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
    
    // Update performance statistics
    impl_->total_generations_++;
    impl_->total_tokens_generated_ += generated_tokens;
    impl_->total_generation_time_ms_ += result.total_time_ms;
    
    // Update average tokens/second
    impl_->average_tokens_per_second_ = impl_->total_tokens_generated_ / (impl_->total_generation_time_ms_ / 1000.0f);
    
    // Track peak performance
    if (result.tokens_per_second > impl_->peak_tokens_per_second_) {
        impl_->peak_tokens_per_second_ = result.tokens_per_second;
    }
    
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
    if (!impl_) {
        return 0; // No memory used if not initialized
    }
    
    size_t total_memory = 0;
    
    // 1. Model tensor weights (use the built-in method)
    const ModelData& model_data = impl_->model_data_;
    total_memory += model_data.total_memory_usage();
    
    // 2. KV-cache memory
    for (const auto& key_tensor : impl_->kv_cache_.key_cache) {
        total_memory += calculate_tensor_memory(key_tensor);
    }
    for (const auto& value_tensor : impl_->kv_cache_.value_cache) {
        total_memory += calculate_tensor_memory(value_tensor);
    }
    
    // 3. Transformer layer internal tensors (if cached)
    if (impl_->token_embeddings_) {
        total_memory += calculate_tensor_memory(*impl_->token_embeddings_);
    }
    if (impl_->output_norm_) {
        total_memory += calculate_tensor_memory(*impl_->output_norm_);
    }
    if (impl_->lm_head_) {
        total_memory += calculate_tensor_memory(*impl_->lm_head_);
    }
    
    // 4. Layer-specific weights (attention and FFN weights)
    for (const auto& layer : impl_->layers_) {
        if (layer.q_proj) total_memory += calculate_tensor_memory(*layer.q_proj);
        if (layer.k_proj) total_memory += calculate_tensor_memory(*layer.k_proj);
        if (layer.v_proj) total_memory += calculate_tensor_memory(*layer.v_proj);
        if (layer.o_proj) total_memory += calculate_tensor_memory(*layer.o_proj);
        if (layer.ffn_gate) total_memory += calculate_tensor_memory(*layer.ffn_gate);
        if (layer.ffn_up) total_memory += calculate_tensor_memory(*layer.ffn_up);
        if (layer.ffn_down) total_memory += calculate_tensor_memory(*layer.ffn_down);
        if (layer.attention_norm) total_memory += calculate_tensor_memory(*layer.attention_norm);
        if (layer.ffn_norm) total_memory += calculate_tensor_memory(*layer.ffn_norm);
    }
    
    // 5. Additional overhead (approximate)
    // - Object instances, metadata, etc.
    size_t overhead = sizeof(InferenceEngineImpl) + 
                     sizeof(*this) + 
                     impl_->layers_.size() * sizeof(TransformerLayer) +
                     1024 * 1024; // 1MB estimated overhead for other allocations
    
    total_memory += overhead;
    
    return total_memory;
}

/**
 * @brief Calculate memory usage of a single tensor.
 * @param tensor The tensor to calculate memory for.
 * @return Memory usage in bytes.
 */
size_t InferenceEngine::calculate_tensor_memory(const core::Tensor& tensor) const {
    size_t element_count = tensor.shape().total_size();
    size_t element_size = 0;
    
    // Calculate bytes per element based on data type
    switch (tensor.dtype()) {
        case core::DataType::kFloat32:
            element_size = 4; // 32 bits = 4 bytes
            break;
        case core::DataType::kFloat16:
            element_size = 2; // 16 bits = 2 bytes
            break;
        case core::DataType::kInt32:
            element_size = 4; // 32 bits = 4 bytes
            break;
        case core::DataType::kInt16:
            element_size = 2; // 16 bits = 2 bytes
            break;
        case core::DataType::kInt8:
        case core::DataType::kUInt8:
            element_size = 1; // 8 bits = 1 byte
            break;
        default:
            element_size = 4; // Default to 4 bytes if unknown
            break;
    }
    
    return element_count * element_size;
}

std::string InferenceEngine::performance_stats() const {
    if (!impl_) {
        return "Performance statistics unavailable (engine not initialized)";
    }
    
    std::ostringstream stats;
    stats << std::fixed << std::setprecision(2);
    
    stats << "=== TurboInfer Performance Statistics ===\n";
    
    // Generation Statistics
    stats << "\n📊 Generation Performance:\n";
    stats << "  Total Generations: " << impl_->total_generations_ << "\n";
    stats << "  Total Tokens Generated: " << impl_->total_tokens_generated_ << "\n";
    stats << "  Total Generation Time: " << impl_->total_generation_time_ms_ << " ms\n";
    
    if (impl_->total_generations_ > 0) {
        float avg_tokens_per_gen = static_cast<float>(impl_->total_tokens_generated_) / impl_->total_generations_;
        float avg_time_per_gen = impl_->total_generation_time_ms_ / impl_->total_generations_;
        stats << "  Average Tokens/Generation: " << avg_tokens_per_gen << "\n";
        stats << "  Average Time/Generation: " << avg_time_per_gen << " ms\n";
    }
    
    // Throughput Statistics
    stats << "\n🚀 Throughput Performance:\n";
    if (impl_->total_generation_time_ms_ > 0) {
        stats << "  Average Speed: " << impl_->average_tokens_per_second_ << " tokens/second\n";
    } else {
        stats << "  Average Speed: N/A (no generations yet)\n";
    }
    stats << "  Peak Speed: " << impl_->peak_tokens_per_second_ << " tokens/second\n";
    
    // Forward Pass Statistics
    stats << "\n⚡ Forward Pass Performance:\n";
    stats << "  Total Forward Passes: " << impl_->total_forward_passes_ << "\n";
    stats << "  Total Forward Time: " << impl_->total_forward_time_ms_ << " ms\n";
    
    if (impl_->total_forward_passes_ > 0) {
        float avg_forward_time = impl_->total_forward_time_ms_ / impl_->total_forward_passes_;
        stats << "  Average Forward Pass Time: " << avg_forward_time << " ms\n";
        stats << "  Forward Passes/Second: " << (1000.0f / avg_forward_time) << "\n";
    }
    
    // Cache Performance
    stats << "\n💾 KV-Cache Performance:\n";
    size_t cache_length = impl_->kv_cache_.current_length;
    size_t cache_capacity = impl_->kv_cache_.max_length;
    float cache_utilization = cache_capacity > 0 ? (static_cast<float>(cache_length) / cache_capacity * 100.0f) : 0.0f;
    
    stats << "  Current Cache Length: " << cache_length << " tokens\n";
    stats << "  Cache Capacity: " << cache_capacity << " tokens\n";
    stats << "  Cache Utilization: " << cache_utilization << "%\n";
    
    // Model Information
    stats << "\n🧠 Model Information:\n";
    stats << "  Architecture: " << model_metadata_.architecture << "\n";
    stats << "  Layers: " << model_metadata_.num_layers << "\n";
    stats << "  Hidden Size: " << model_metadata_.hidden_size << "\n";
    stats << "  Attention Heads: " << model_metadata_.num_heads << "\n";
    stats << "  Vocabulary Size: " << model_metadata_.vocab_size << "\n";
    
    // Efficiency Metrics
    stats << "\n📈 Efficiency Metrics:\n";
    if (impl_->total_tokens_generated_ > 0 && impl_->total_generation_time_ms_ > 0) {
        float efficiency_score = impl_->average_tokens_per_second_ / 1000.0f; // Normalized score
        std::string efficiency_rating;
        if (efficiency_score > 1.0f) efficiency_rating = "Excellent";
        else if (efficiency_score > 0.5f) efficiency_rating = "Good";
        else if (efficiency_score > 0.2f) efficiency_rating = "Fair";
        else efficiency_rating = "Needs Optimization";
        
        stats << "  Efficiency Score: " << efficiency_score << "\n";
        stats << "  Performance Rating: " << efficiency_rating << "\n";
    } else {
        stats << "  Efficiency Score: N/A (insufficient data)\n";
        stats << "  Performance Rating: N/A\n";
    }
    
    // Memory Usage Estimate
    stats << "\n🔢 Resource Usage:\n";
    size_t model_size_mb = static_cast<size_t>(memory_usage() / (1024 * 1024));
    stats << "  Estimated Memory Usage: " << model_size_mb << " MB\n";
    
    if (impl_->total_forward_passes_ > 0) {
        float compute_intensity = static_cast<float>(impl_->total_tokens_generated_) / impl_->total_forward_passes_;
        stats << "  Compute Intensity: " << compute_intensity << " tokens/forward_pass\n";
    }
    
    stats << "\n=== End Performance Statistics ===";
    
    return stats.str();
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
    // Create a copy of logits to modify
    core::Tensor filtered_logits = logits;
    float* logit_data = filtered_logits.data_ptr<float>();
    size_t vocab_size = logits.shape().size(logits.shape().ndim() - 1);
    
    if (k >= vocab_size) {
        return filtered_logits; // No filtering needed
    }
    
    // Create pairs of (logit_value, index) for sorting
    std::vector<std::pair<float, size_t>> logit_pairs;
    logit_pairs.reserve(vocab_size);
    
    for (size_t i = 0; i < vocab_size; ++i) {
        logit_pairs.emplace_back(logit_data[i], i);
    }
    
    // Sort by logit value in descending order
    std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + k, logit_pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Set all but top-k logits to negative infinity
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<bool> is_top_k(vocab_size, false);
    
    for (size_t i = 0; i < k; ++i) {
        is_top_k[logit_pairs[i].second] = true;
    }
    
    for (size_t i = 0; i < vocab_size; ++i) {
        if (!is_top_k[i]) {
            logit_data[i] = neg_inf;
        }
    }
    
    return filtered_logits;
}

core::Tensor InferenceEngine::apply_top_p(const core::Tensor& logits, float p) {
    if (p >= 1.0f) {
        return logits; // No filtering needed
    }
    
    // Create a copy of logits to modify
    core::Tensor filtered_logits = logits;
    float* logit_data = filtered_logits.data_ptr<float>();
    size_t vocab_size = logits.shape().size(logits.shape().ndim() - 1);
    
    // Convert logits to probabilities using softmax
    std::vector<float> probs(vocab_size);
    
    // Find max for numerical stability
    float max_logit = *std::max_element(logit_data, logit_data + vocab_size);
    
    // Compute softmax
    float sum_exp = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(logit_data[i] - max_logit);
        sum_exp += probs[i];
    }
    
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] /= sum_exp;
    }
    
    // Create pairs of (probability, index) and sort by probability
    std::vector<std::pair<float, size_t>> prob_pairs;
    prob_pairs.reserve(vocab_size);
    
    for (size_t i = 0; i < vocab_size; ++i) {
        prob_pairs.emplace_back(probs[i], i);
    }
    
    std::sort(prob_pairs.begin(), prob_pairs.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Find nucleus cutoff point
    float cumulative_prob = 0.0f;
    std::vector<bool> in_nucleus(vocab_size, false);
    
    for (const auto& pair : prob_pairs) {
        cumulative_prob += pair.first;
        in_nucleus[pair.second] = true;
        
        if (cumulative_prob >= p) {
            break;
        }
    }
    
    // Set logits outside nucleus to negative infinity
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab_size; ++i) {
        if (!in_nucleus[i]) {
            logit_data[i] = neg_inf;
        }
    }
    
    return filtered_logits;
}

std::vector<float> InferenceEngine::softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    
    // Find maximum for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (auto& prob : probs) {
            prob /= sum;
        }
    }
    
    return probs;
}

std::vector<float> InferenceEngine::apply_top_k_filtering(const std::vector<float>& probs, size_t k) {
    std::vector<float> filtered_probs = probs;
    
    if (k >= probs.size()) {
        return filtered_probs; // No filtering needed
    }
    
    // Create index-probability pairs
    std::vector<std::pair<float, size_t>> prob_indices;
    for (size_t i = 0; i < probs.size(); ++i) {
        prob_indices.emplace_back(probs[i], i);
    }
    
    // Sort by probability (descending)
    std::sort(prob_indices.begin(), prob_indices.end(), 
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Zero out probabilities outside top-k
    for (size_t i = k; i < prob_indices.size(); ++i) {
        filtered_probs[prob_indices[i].second] = 0.0f;
    }
    
    // Renormalize
    float sum = 0.0f;
    for (float prob : filtered_probs) {
        sum += prob;
    }
    
    if (sum > 0.0f) {
        for (auto& prob : filtered_probs) {
            prob /= sum;
        }
    }
    
    return filtered_probs;
}

std::vector<float> InferenceEngine::apply_top_p_filtering(const std::vector<float>& probs, float p) {
    std::vector<float> filtered_probs = probs;
    
    if (p >= 1.0f) {
        return filtered_probs; // No filtering needed
    }
    
    // Create index-probability pairs and sort by probability (descending)
    std::vector<std::pair<float, size_t>> prob_indices;
    for (size_t i = 0; i < probs.size(); ++i) {
        prob_indices.emplace_back(probs[i], i);
    }
    
    std::sort(prob_indices.begin(), prob_indices.end(), 
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Find nucleus (top-p set)
    float cumulative_prob = 0.0f;
    std::vector<bool> in_nucleus(probs.size(), false);
    
    for (const auto& pair : prob_indices) {
        float prob = pair.first;
        size_t index = pair.second;
        
        cumulative_prob += prob;
        in_nucleus[index] = true;
        
        if (cumulative_prob >= p) {
            break;
        }
    }
    
    // Zero out probabilities outside nucleus
    for (size_t i = 0; i < probs.size(); ++i) {
        if (!in_nucleus[i]) {
            filtered_probs[i] = 0.0f;
        }
    }
    
    // Renormalize
    float sum = 0.0f;
    for (float prob : filtered_probs) {
        sum += prob;
    }
    
    if (sum > 0.0f) {
        for (auto& prob : filtered_probs) {
            prob /= sum;
        }
    }
    
    return filtered_probs;
}

std::vector<InferenceEngine::BeamCandidate> InferenceEngine::beam_search_decode(
    const std::vector<int>& input_tokens,
    size_t max_new_tokens,
    size_t beam_size) {
    
    if (beam_size == 0) {
        throw std::invalid_argument("Beam size must be greater than 0");
    }
    
    // Priority queue for beam candidates (max-heap by log probability)
    auto compare = [](const BeamCandidate& a, const BeamCandidate& b) {
        return a.log_prob < b.log_prob; // For max-heap behavior
    };
    std::priority_queue<BeamCandidate, std::vector<BeamCandidate>, decltype(compare)> beam(compare);
    
    // Initialize beam with input sequence
    BeamCandidate initial_candidate;
    initial_candidate.tokens = input_tokens;
    initial_candidate.log_prob = 0.0f;
    initial_candidate.finished = false;
    beam.push(initial_candidate);
    
    std::vector<BeamCandidate> finished_sequences;
    
    // Generate tokens iteratively
    for (size_t step = 0; step < max_new_tokens; ++step) {
        std::vector<BeamCandidate> current_candidates;
        
        // Extract all current beam candidates
        while (!beam.empty()) {
            current_candidates.push_back(beam.top());
            beam.pop();
        }
        
        // If no candidates left, break
        if (current_candidates.empty()) {
            break;
        }
        
        std::vector<BeamCandidate> next_candidates;
        
        // Expand each candidate
        for (const auto& candidate : current_candidates) {
            if (candidate.finished) {
                finished_sequences.push_back(candidate);
                continue;
            }
            
            // Get next token logits for this candidate
            auto logits_tensor = forward_pass(candidate.tokens);
            
            // Extract logits data (assume last token's logits)
            const float* logits_data = logits_tensor.data_ptr<float>();
            size_t vocab_size = logits_tensor.shape().total_size() / logits_tensor.shape().size(0); // Total size / batch size
            
            // Convert to vector for easier manipulation
            std::vector<float> logits(logits_data, logits_data + vocab_size);
            
            // Apply temperature scaling if configured
            if (config_.temperature != 1.0f) {
                for (auto& logit : logits) {
                    logit /= config_.temperature;
                }
            }
            
            // Convert logits to probabilities
            std::vector<float> probs = softmax(logits);
            
            // Apply top-k filtering if configured
            if (config_.top_k > 0 && config_.top_k < probs.size()) {
                probs = apply_top_k_filtering(probs, config_.top_k);
            }
            
            // Apply top-p (nucleus) filtering if configured
            if (config_.top_p < 1.0f) {
                probs = apply_top_p_filtering(probs, config_.top_p);
            }
            
            // Get top beam_size candidates for expansion
            std::vector<std::pair<float, int>> prob_token_pairs;
            for (size_t i = 0; i < probs.size(); ++i) {
                if (probs[i] > 0.0f) {
                    prob_token_pairs.emplace_back(probs[i], static_cast<int>(i));
                }
            }
            
            // Sort by probability (descending)
            std::sort(prob_token_pairs.begin(), prob_token_pairs.end(), 
                     [](const auto& a, const auto& b) { return a.first > b.first; });
            
            // Take top candidates for beam expansion
            size_t expansion_count = std::min(beam_size, prob_token_pairs.size());
            for (size_t i = 0; i < expansion_count; ++i) {
                float prob = prob_token_pairs[i].first;
                int token = prob_token_pairs[i].second;
                
                if (prob <= 0.0f) continue; // Skip zero probability tokens
                
                BeamCandidate new_candidate = candidate;
                new_candidate.tokens.push_back(token);
                new_candidate.log_prob += std::log(prob);
                
                // Check if sequence is finished (EOS token or max length)
                new_candidate.finished = (token == config_.eos_token_id) || 
                                       (new_candidate.tokens.size() >= input_tokens.size() + max_new_tokens);
                
                next_candidates.push_back(new_candidate);
            }
        }
        
        // Apply length normalization and select best candidates
        for (auto& candidate : next_candidates) {
            // Length normalization: divide by sequence length^alpha
            float length_penalty = std::pow(static_cast<float>(candidate.tokens.size()), config_.length_penalty);
            candidate.normalized_score = candidate.log_prob / length_penalty;
        }
        
        // Sort by normalized score and keep top beam_size candidates
        std::sort(next_candidates.begin(), next_candidates.end(), 
                 [](const auto& a, const auto& b) { 
                     return a.normalized_score > b.normalized_score; 
                 });
        
        // Add top candidates back to beam
        size_t candidates_to_keep = std::min(beam_size, next_candidates.size());
        for (size_t i = 0; i < candidates_to_keep; ++i) {
            if (next_candidates[i].finished) {
                finished_sequences.push_back(next_candidates[i]);
            } else {
                beam.push(next_candidates[i]);
            }
        }
        
        // Early stopping: if we have enough finished sequences
        if (finished_sequences.size() >= beam_size) {
            break;
        }
    }
    
    // Collect remaining beam candidates as finished
    while (!beam.empty()) {
        BeamCandidate candidate = beam.top();
        beam.pop();
        candidate.finished = true;
        finished_sequences.push_back(candidate);
    }
    
    // Sort final results by normalized score
    std::sort(finished_sequences.begin(), finished_sequences.end(), 
             [](const auto& a, const auto& b) { 
                 return a.normalized_score > b.normalized_score; 
             });
    
    // Return top beam_size results
    size_t result_count = std::min(beam_size, finished_sequences.size());
    return std::vector<BeamCandidate>(finished_sequences.begin(), 
                                     finished_sequences.begin() + result_count);
}

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
