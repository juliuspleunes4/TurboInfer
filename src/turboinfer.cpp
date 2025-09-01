/**
 * @file turboinfer.cpp
 * @brief Implementation of main TurboInfer library functions.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/turboinfer.hpp"
#include "turboinfer/util/logging.hpp"
#include "turboinfer/model/inference_engine.hpp"
#include <chrono>
#include <sstream>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace turboinfer {

namespace {
    bool g_initialized = false;
    
    // Global tokenizer cache to avoid creating new engines for each call
    std::unordered_map<std::string, std::shared_ptr<model::InferenceEngine>> g_tokenizer_cache;
    std::mutex g_tokenizer_cache_mutex;
}

const char* build_info() {
    std::ostringstream info;
    info << "TurboInfer " << Version::kString;
    info << " (built " << __DATE__ << " " << __TIME__ << ")";
    info << " C++" << __cplusplus;
    
#ifdef TURBOINFER_OPENMP_ENABLED
    info << " +OpenMP";
#endif

#ifdef TURBOINFER_SIMD_ENABLED
    info << " +SIMD";
#endif

#if defined(_WIN32)
    info << " Windows";
#elif defined(__linux__)
    info << " Linux";
#elif defined(__APPLE__)
    info << " macOS";
#endif

    static std::string build_info_str = info.str();
    return build_info_str.c_str();
}

bool initialize(bool enable_logging) {
    if (g_initialized) {
        return true;
    }

    try {
        // Initialize logging
        if (enable_logging) {
            auto& logger = util::Logger::instance();
            logger.set_level(util::LogLevel::kInfo);
            logger.set_console_output(true);
            logger.set_timestamp_enabled(true);
        }

        TURBOINFER_LOG_INFO() << "Initializing TurboInfer " << Version::kString;
        TURBOINFER_LOG_INFO() << "Build info: " << build_info();

        // Initialize any global systems here
        // - Hardware detection
        // - Memory pools
        // - Thread pools
        // - SIMD capability detection
        
        TURBOINFER_LOG_INFO() << "TurboInfer initialization completed successfully";
        
        g_initialized = true;
        return true;

    } catch (const std::exception& e) {
        if (enable_logging) {
            TURBOINFER_LOG_ERROR() << "Failed to initialize TurboInfer: " << e.what();
        }
        return false;
    } catch (...) {
        if (enable_logging) {
            TURBOINFER_LOG_ERROR() << "Failed to initialize TurboInfer: unknown error";
        }
        return false;
    }
}

void shutdown() {
    if (!g_initialized) {
        return;
    }

    TURBOINFER_LOG_INFO() << "Shutting down TurboInfer";
    
    // Clean up global systems
    {
        std::lock_guard<std::mutex> lock(g_tokenizer_cache_mutex);
        size_t cached_tokenizers = g_tokenizer_cache.size();
        g_tokenizer_cache.clear();
        if (cached_tokenizers > 0) {
            TURBOINFER_LOG_INFO() << "Cleared " << cached_tokenizers << " cached tokenizers";
        }
    }
    
    // Flush logging
    util::Logger::instance().flush();
    
    g_initialized = false;
}

bool is_initialized() {
    return g_initialized;
}

/**
 * @brief Gets or creates a cached tokenizer for the given model path.
 * @param model_path Path to the model file.
 * @return Shared pointer to the inference engine (used as tokenizer).
 */
std::shared_ptr<model::InferenceEngine> get_cached_tokenizer(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(g_tokenizer_cache_mutex);
    
    auto it = g_tokenizer_cache.find(model_path);
    if (it != g_tokenizer_cache.end()) {
        return it->second;
    }
    
    // Create new tokenizer and cache it
    try {
        auto tokenizer = std::make_shared<model::InferenceEngine>(model_path);
        g_tokenizer_cache[model_path] = tokenizer;
        return tokenizer;
    } catch (const std::exception& e) {
        TURBOINFER_LOG_ERROR() << "Failed to create tokenizer for " << model_path << ": " << e.what();
        throw;
    }
}

std::vector<int> tokenize(const std::string& text, const std::string& model_path) {
    if (!g_initialized) {
        TURBOINFER_LOG_WARNING() << "TurboInfer not initialized, call turboinfer::initialize() first";
    }
    
    try {
        auto tokenizer = get_cached_tokenizer(model_path);
        return tokenizer->encode(text);
    } catch (const std::exception& e) {
        TURBOINFER_LOG_ERROR() << "Tokenization failed: " << e.what();
        return {}; // Return empty vector on error
    }
}

std::string detokenize(const std::vector<int>& tokens, const std::string& model_path) {
    if (!g_initialized) {
        TURBOINFER_LOG_WARNING() << "TurboInfer not initialized, call turboinfer::initialize() first";
    }
    
    try {
        auto tokenizer = get_cached_tokenizer(model_path);
        return tokenizer->decode(tokens);
    } catch (const std::exception& e) {
        TURBOINFER_LOG_ERROR() << "Detokenization failed: " << e.what();
        return ""; // Return empty string on error
    }
}

} // namespace turboinfer
