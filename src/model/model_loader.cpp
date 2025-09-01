/**
 * @file model_loader.cpp
 * @brief Implementation of model loading utilities (placeholder).
 * @author J.J.G. Pleunes
 */

#include "turboinfer/model/model_loader.hpp"
#include <stdexcept>
#include <filesystem>
#include <vector>
#include <string>
#include <cstddef>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cstdio>

// GGUF format constants and structures
namespace {
    constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" in little-endian
    constexpr uint32_t GGUF_VERSION = 3;         // Current GGUF version
    
    // GGUF data types
    enum class GGUFType : uint32_t {
        UINT8 = 0,
        INT8 = 1,
        UINT16 = 2,
        INT16 = 3,
        UINT32 = 4,
        INT32 = 5,
        FLOAT32 = 6,
        BOOL = 7,
        STRING = 8,
        ARRAY = 9,
        UINT64 = 10,
        INT64 = 11,
        FLOAT64 = 12,
    };
    
    struct GGUFHeader {
        uint32_t magic;
        uint32_t version;
        uint64_t tensor_count;
        uint64_t metadata_kv_count;
    };
    
    // Helper functions for reading GGUF data
    std::string read_gguf_string(std::ifstream& file) {
        uint64_t length;
        file.read(reinterpret_cast<char*>(&length), sizeof(length));
        
        if (length > 1024 * 1024) { // Sanity check: max 1MB strings
            throw std::runtime_error("GGUF string too long");
        }
        
        std::string result(length, '\0');
        file.read(result.data(), static_cast<std::streamsize>(length));
        
        if (!file.good()) {
            throw std::runtime_error("Failed to read GGUF string");
        }
        
        return result;
    }
    
    GGUFType read_gguf_value_type(std::ifstream& file) {
        uint32_t type;
        file.read(reinterpret_cast<char*>(&type), sizeof(type));
        
        if (!file.good()) {
            throw std::runtime_error("Failed to read GGUF value type");
        }
        
        return static_cast<GGUFType>(type);
    }
    
    std::string read_gguf_value(std::ifstream& file, GGUFType type) {
        switch (type) {
            case GGUFType::UINT8: {
                uint8_t value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                return std::to_string(value);
            }
            case GGUFType::INT8: {
                int8_t value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                return std::to_string(value);
            }
            case GGUFType::UINT16: {
                uint16_t value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                return std::to_string(value);
            }
            case GGUFType::INT16: {
                int16_t value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                return std::to_string(value);
            }
            case GGUFType::UINT32: {
                uint32_t value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                return std::to_string(value);
            }
            case GGUFType::INT32: {
                int32_t value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                return std::to_string(value);
            }
            case GGUFType::UINT64: {
                uint64_t value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                return std::to_string(value);
            }
            case GGUFType::INT64: {
                int64_t value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                return std::to_string(value);
            }
            case GGUFType::FLOAT32: {
                float value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                return std::to_string(value);
            }
            case GGUFType::FLOAT64: {
                double value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                return std::to_string(value);
            }
            case GGUFType::BOOL: {
                uint8_t value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                return value ? "true" : "false";
            }
            case GGUFType::STRING: {
                return read_gguf_string(file);
            }
            case GGUFType::ARRAY: {
                // For arrays, we'll skip for now and return empty string
                // Full array parsing would be complex and not needed for basic metadata
                uint32_t array_type;
                uint64_t array_length;
                file.read(reinterpret_cast<char*>(&array_type), sizeof(array_type));
                file.read(reinterpret_cast<char*>(&array_length), sizeof(array_length));
                
                // Skip array data for now
                file.seekg(static_cast<std::streamoff>(array_length * 8), std::ios::cur); // Rough estimate
                return "[array]";
            }
            default:
                throw std::runtime_error("Unsupported GGUF value type: " + std::to_string(static_cast<uint32_t>(type)));
        }
    }
}

namespace turboinfer {
namespace model {

/**
 * @brief Converts a GGUF tensor type to TurboInfer DataType.
 * @param gguf_type The GGUF tensor type identifier.
 * @return The corresponding TurboInfer DataType.
 */
core::DataType convert_gguf_type_to_turboinfer(uint32_t gguf_type) {
    // GGUF tensor types are different from metadata types
    // Common GGUF tensor types:
    // 0 = F32, 1 = F16, 2 = Q4_0, 3 = Q4_1, etc.
    switch (gguf_type) {
        case 0: return core::DataType::kFloat32;  // F32
        case 1: return core::DataType::kFloat16;  // F16
        // For quantized types, we'll use Float32 for now
        case 2: return core::DataType::kFloat32;  // Q4_0 -> F32 (dequantized)
        case 3: return core::DataType::kFloat32;  // Q4_1 -> F32 (dequantized)
        case 6: return core::DataType::kFloat32;  // Q5_0 -> F32 (dequantized)
        case 7: return core::DataType::kFloat32;  // Q5_1 -> F32 (dequantized)
        case 8: return core::DataType::kFloat32;  // Q8_0 -> F32 (dequantized)
        default:
            // Default to Float32 for unknown types
            return core::DataType::kFloat32;
    }
}

// ModelData implementation

const core::Tensor* ModelData::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

core::Tensor* ModelData::get_tensor(const std::string& name) {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

void ModelData::add_tensor(const std::string& name, core::Tensor tensor) {
    tensors_.insert_or_assign(name, std::move(tensor));
}

std::vector<std::string> ModelData::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        names.push_back(pair.first);
    }
    return names;
}

bool ModelData::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

size_t ModelData::total_memory_usage() const {
    size_t total = 0;
    for (const auto& pair : tensors_) {
        total += pair.second.byte_size();
    }
    return total;
}

std::string ModelData::get_model_summary() const {
    std::string summary;
    summary += "Model Summary:\n";
    summary += "  Name: " + metadata_.name + "\n";
    summary += "  Architecture: " + metadata_.architecture + "\n";
    summary += "  Version: " + metadata_.version + "\n";
    summary += "  Vocabulary Size: " + std::to_string(metadata_.vocab_size) + "\n";
    summary += "  Hidden Size: " + std::to_string(metadata_.hidden_size) + "\n";
    summary += "  Layers: " + std::to_string(metadata_.num_layers) + "\n";
    summary += "  Attention Heads: " + std::to_string(metadata_.num_heads) + "\n";
    summary += "  Intermediate Size: " + std::to_string(metadata_.intermediate_size) + "\n";
    summary += "  RoPE Theta: " + std::to_string(metadata_.rope_theta) + "\n";
    summary += "  Number of Tensors: " + std::to_string(num_tensors()) + "\n";
    summary += "  Memory Usage: " + get_memory_usage_string() + "\n";
    
    if (!metadata_.extra_params.empty()) {
        summary += "  Extra Parameters:\n";
        for (const auto& param : metadata_.extra_params) {
            summary += "    " + param.first + ": " + param.second + "\n";
        }
    }
    
    return summary;
}

bool ModelData::validate() const {
    // Check if model has essential metadata
    if (metadata_.name.empty()) return false;
    if (metadata_.architecture.empty()) return false;
    
    // Check if model has any tensors
    if (tensors_.empty()) return false;
    
    // Check tensor consistency
    for (const auto& [name, tensor] : tensors_) {
        const auto& shape = tensor.shape();
        if (shape.dimensions().empty()) return false;
        
        // Check for zero dimensions
        for (size_t dim : shape.dimensions()) {
            if (dim == 0) return false;
        }
    }
    
    // Check for reasonable metadata values
    if (metadata_.vocab_size > 0 && metadata_.vocab_size > 1000000) {
        return false; // Vocab size too large
    }
    
    if (metadata_.hidden_size > 0 && metadata_.hidden_size > 32768) {
        return false; // Hidden size too large
    }
    
    return true;
}

std::string ModelData::get_memory_usage_string() const {
    size_t bytes = total_memory_usage();
    
    if (bytes == 0) return "0 B";
    
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    char buffer[32];
    if (unit_index == 0) {
        snprintf(buffer, sizeof(buffer), "%.0f %s", size, units[unit_index]);
    } else {
        snprintf(buffer, sizeof(buffer), "%.1f %s", size, units[unit_index]);
    }
    
    return std::string(buffer);
}

void ModelData::set_config_param(const std::string& key, const std::string& value) {
    metadata_.extra_params[key] = value;
}

std::string ModelData::get_config_param(const std::string& key, const std::string& default_value) const {
    auto it = metadata_.extra_params.find(key);
    if (it != metadata_.extra_params.end()) {
        return it->second;
    }
    return default_value;
}

// SafeTensors helper functions
namespace {
    /**
     * @struct SafeTensorInfo
     * @brief Information about a tensor in SafeTensors format.
     */
    struct SafeTensorInfo {
        std::string dtype;
        std::vector<size_t> shape;
        std::vector<size_t> data_offsets; // [start, end]
    };
    
    /**
     * @brief Converts SafeTensors data type string to TurboInfer DataType.
     * @param dtype SafeTensors dtype string (e.g., "F32", "I32", "U8").
     * @return Corresponding TurboInfer DataType.
     */
    core::DataType convert_safetensors_dtype(const std::string& dtype) {
        if (dtype == "F32") return core::DataType::kFloat32;
        if (dtype == "F16") return core::DataType::kFloat32; // Convert F16 to F32 for now
        if (dtype == "I32") return core::DataType::kInt32;
        if (dtype == "I8")  return core::DataType::kInt32;  // Convert I8 to I32
        if (dtype == "U8")  return core::DataType::kInt32;  // Convert U8 to I32
        if (dtype == "BOOL") return core::DataType::kInt32; // Convert BOOL to I32
        
        throw std::runtime_error("Unsupported SafeTensors dtype: " + dtype);
    }
    
    /**
     * @brief Simple JSON parser for SafeTensors format.
     * @param json_str JSON string to parse.
     * @return Map of tensor name to tensor information.
     */
    std::unordered_map<std::string, SafeTensorInfo> parse_safetensors_json(const std::string& json_str) {
        std::unordered_map<std::string, SafeTensorInfo> result;
        
        // Find the start and end of the JSON object
        size_t start = json_str.find('{');
        size_t end = json_str.rfind('}');
        
        if (start == std::string::npos || end == std::string::npos) {
            throw std::runtime_error("Invalid JSON in SafeTensors header");
        }
        
        // Simple state machine to parse JSON
        size_t pos = start + 1;
        
        while (pos < end) {
            // Skip whitespace
            while (pos < end && std::isspace(json_str[pos])) pos++;
            if (pos >= end) break;
            
            // Parse tensor name (key)
            if (json_str[pos] != '"') {
                throw std::runtime_error("Expected quote for tensor name in SafeTensors JSON");
            }
            pos++; // Skip opening quote
            
            size_t name_start = pos;
            while (pos < end && json_str[pos] != '"') pos++;
            if (pos >= end) {
                throw std::runtime_error("Unterminated tensor name in SafeTensors JSON");
            }
            
            std::string tensor_name = json_str.substr(name_start, pos - name_start);
            pos++; // Skip closing quote
            
            // Skip whitespace and colon
            while (pos < end && std::isspace(json_str[pos])) pos++;
            if (pos >= end || json_str[pos] != ':') {
                throw std::runtime_error("Expected colon after tensor name in SafeTensors JSON");
            }
            pos++; // Skip colon
            
            // Skip whitespace
            while (pos < end && std::isspace(json_str[pos])) pos++;
            
            // Parse tensor info object
            if (json_str[pos] != '{') {
                throw std::runtime_error("Expected object for tensor info in SafeTensors JSON");
            }
            pos++; // Skip opening brace
            
            SafeTensorInfo info;
            
            // Parse tensor object properties
            while (pos < end && json_str[pos] != '}') {
                // Skip whitespace
                while (pos < end && std::isspace(json_str[pos])) pos++;
                if (pos >= end || json_str[pos] == '}') break;
                
                // Parse property name
                if (json_str[pos] != '"') {
                    throw std::runtime_error("Expected quote for property name in SafeTensors JSON");
                }
                pos++; // Skip opening quote
                
                size_t prop_start = pos;
                while (pos < end && json_str[pos] != '"') pos++;
                if (pos >= end) {
                    throw std::runtime_error("Unterminated property name in SafeTensors JSON");
                }
                
                std::string prop_name = json_str.substr(prop_start, pos - prop_start);
                pos++; // Skip closing quote
                
                // Skip whitespace and colon
                while (pos < end && std::isspace(json_str[pos])) pos++;
                if (pos >= end || json_str[pos] != ':') {
                    throw std::runtime_error("Expected colon after property name in SafeTensors JSON");
                }
                pos++; // Skip colon
                
                // Skip whitespace
                while (pos < end && std::isspace(json_str[pos])) pos++;
                
                // Parse property value based on property name
                if (prop_name == "dtype") {
                    // Parse string value
                    if (json_str[pos] != '"') {
                        throw std::runtime_error("Expected string for dtype in SafeTensors JSON");
                    }
                    pos++; // Skip opening quote
                    
                    size_t dtype_start = pos;
                    while (pos < end && json_str[pos] != '"') pos++;
                    if (pos >= end) {
                        throw std::runtime_error("Unterminated dtype string in SafeTensors JSON");
                    }
                    
                    info.dtype = json_str.substr(dtype_start, pos - dtype_start);
                    pos++; // Skip closing quote
                    
                } else if (prop_name == "shape") {
                    // Parse array of numbers
                    if (json_str[pos] != '[') {
                        throw std::runtime_error("Expected array for shape in SafeTensors JSON");
                    }
                    pos++; // Skip opening bracket
                    
                    while (pos < end && json_str[pos] != ']') {
                        // Skip whitespace
                        while (pos < end && std::isspace(json_str[pos])) pos++;
                        if (pos >= end || json_str[pos] == ']') break;
                        
                        // Parse number
                        size_t num_start = pos;
                        while (pos < end && std::isdigit(json_str[pos])) pos++;
                        
                        if (pos == num_start) {
                            throw std::runtime_error("Expected number in shape array in SafeTensors JSON");
                        }
                        
                        size_t dim = std::stoull(json_str.substr(num_start, pos - num_start));
                        info.shape.push_back(dim);
                        
                        // Skip whitespace
                        while (pos < end && std::isspace(json_str[pos])) pos++;
                        
                        // Skip comma if present
                        if (pos < end && json_str[pos] == ',') {
                            pos++;
                        }
                    }
                    
                    if (pos >= end || json_str[pos] != ']') {
                        throw std::runtime_error("Expected closing bracket for shape array in SafeTensors JSON");
                    }
                    pos++; // Skip closing bracket
                    
                } else if (prop_name == "data_offsets") {
                    // Parse array of two numbers [start, end]
                    if (json_str[pos] != '[') {
                        throw std::runtime_error("Expected array for data_offsets in SafeTensors JSON");
                    }
                    pos++; // Skip opening bracket
                    
                    for (int i = 0; i < 2; i++) {
                        // Skip whitespace
                        while (pos < end && std::isspace(json_str[pos])) pos++;
                        
                        // Parse number
                        size_t num_start = pos;
                        while (pos < end && std::isdigit(json_str[pos])) pos++;
                        
                        if (pos == num_start) {
                            throw std::runtime_error("Expected number in data_offsets array in SafeTensors JSON");
                        }
                        
                        size_t offset = std::stoull(json_str.substr(num_start, pos - num_start));
                        info.data_offsets.push_back(offset);
                        
                        // Skip whitespace
                        while (pos < end && std::isspace(json_str[pos])) pos++;
                        
                        // Skip comma if present and not last element
                        if (i == 0 && pos < end && json_str[pos] == ',') {
                            pos++;
                        }
                    }
                    
                    if (pos >= end || json_str[pos] != ']') {
                        throw std::runtime_error("Expected closing bracket for data_offsets array in SafeTensors JSON");
                    }
                    pos++; // Skip closing bracket
                }
                
                // Skip whitespace
                while (pos < end && std::isspace(json_str[pos])) pos++;
                
                // Skip comma if present
                if (pos < end && json_str[pos] == ',') {
                    pos++;
                }
            }
            
            if (pos >= end || json_str[pos] != '}') {
                throw std::runtime_error("Expected closing brace for tensor info in SafeTensors JSON");
            }
            pos++; // Skip closing brace
            
            // Add tensor info to result
            result[tensor_name] = info;
            
            // Skip whitespace
            while (pos < end && std::isspace(json_str[pos])) pos++;
            
            // Skip comma if present
            if (pos < end && json_str[pos] == ',') {
                pos++;
            }
        }
        
        return result;
    }
} // namespace

// ModelLoader implementation

ModelData ModelLoader::load(const std::string& file_path) {
    ModelFormat format = detect_format(file_path);
    return load(file_path, format);
}

ModelData ModelLoader::load(const std::string& file_path, ModelFormat format) {
    if (!validate_file(file_path)) {
        throw std::runtime_error("Invalid or inaccessible file: " + file_path);
    }

    switch (format) {
        case ModelFormat::kGGUF:
            return load_gguf(file_path);
        case ModelFormat::kSafeTensors:
            return load_safetensors(file_path);
        case ModelFormat::kPyTorch:
            return load_pytorch(file_path);
        case ModelFormat::kONNX:
            return load_onnx(file_path);
        default:
            throw std::runtime_error("Unsupported model format");
    }
}

ModelFormat ModelLoader::detect_format(const std::string& file_path) {
    std::filesystem::path path(file_path);
    std::string extension = path.extension().string();
    
    if (extension == ".gguf") return ModelFormat::kGGUF;
    if (extension == ".safetensors") return ModelFormat::kSafeTensors;
    if (extension == ".pt" || extension == ".pth") return ModelFormat::kPyTorch;
    if (extension == ".onnx") return ModelFormat::kONNX;
    
    throw std::runtime_error("Cannot detect model format from file extension: " + extension);
}

bool ModelLoader::validate_file(const std::string& file_path) {
    std::filesystem::path path(file_path);
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

ModelMetadata ModelLoader::get_model_info(const std::string& file_path) {
    if (!validate_file(file_path)) {
        throw std::runtime_error("Model file does not exist or is not accessible: " + file_path);
    }
    
    ModelFormat format = detect_format(file_path);
    ModelMetadata metadata;
    
    try {
        switch (format) {
            case ModelFormat::kGGUF: {
                // For GGUF files, we can extract metadata from the header
                std::ifstream file(file_path, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Could not open GGUF file for metadata reading");
                }
                
                // Read GGUF header to extract metadata
                char magic[4];
                file.read(magic, 4);
                if (std::string(magic, 4) != "GGUF") {
                    throw std::runtime_error("Invalid GGUF magic number");
                }
                
                uint32_t version;
                file.read(reinterpret_cast<char*>(&version), sizeof(version));
                
                uint64_t tensor_count, metadata_kv_count;
                file.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
                file.read(reinterpret_cast<char*>(&metadata_kv_count), sizeof(metadata_kv_count));
                
                // Set basic info
                metadata.name = std::filesystem::path(file_path).stem().string();
                metadata.architecture = "GGUF";
                metadata.version = std::to_string(version);
                
                // Set default values that would be extracted from metadata
                metadata.vocab_size = 32000;
                metadata.hidden_size = 4096;
                metadata.num_layers = 32;
                metadata.num_heads = 32;
                metadata.intermediate_size = 11008;
                metadata.rope_theta = 10000.0f;
                
                file.close();
                break;
            }
            case ModelFormat::kSafeTensors: {
                // Validate SafeTensors file by trying to read its header
                std::ifstream file(file_path, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Could not open SafeTensors file for metadata reading");
                }
                
                // Try to read the header size (first 8 bytes)
                uint64_t header_size;
                file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
                if (!file.good()) {
                    throw std::runtime_error("Failed to read SafeTensors header size - invalid file format");
                }
                
                // Check if header size is reasonable (between 1 and 100MB)
                if (header_size == 0 || header_size > 100 * 1024 * 1024) {
                    throw std::runtime_error("Invalid SafeTensors header size - corrupt or invalid file");
                }
                
                // Try to read at least the beginning of the header to validate it's JSON
                size_t test_read_size = std::min(header_size, static_cast<uint64_t>(1024));
                std::vector<char> header_test(test_read_size);
                file.read(header_test.data(), test_read_size);
                if (!file.good()) {
                    throw std::runtime_error("Failed to read SafeTensors header - invalid file format");
                }
                
                // Check if it looks like JSON (starts with '{')
                std::string header_start(header_test.begin(), header_test.end());
                if (header_start.empty() || header_start[0] != '{') {
                    throw std::runtime_error("Invalid SafeTensors header - not valid JSON format");
                }
                
                metadata.name = std::filesystem::path(file_path).stem().string();
                metadata.architecture = "SafeTensors";
                metadata.version = "1.0.0";
                // SafeTensors doesn't have built-in metadata, so use defaults
                metadata.vocab_size = 32000;
                metadata.hidden_size = 4096;
                metadata.num_layers = 32;
                metadata.num_heads = 32;
                metadata.intermediate_size = 11008;
                metadata.rope_theta = 10000.0f;
                
                file.close();
                break;
            }
            default: {
                // For other formats, provide basic information
                metadata.name = std::filesystem::path(file_path).stem().string();
                metadata.architecture = "unknown";
                metadata.version = "1.0.0";
                metadata.vocab_size = 32000;
                metadata.hidden_size = 4096;
                metadata.num_layers = 32;
                metadata.num_heads = 32;
                metadata.intermediate_size = 11008;
                metadata.rope_theta = 10000.0f;
                break;
            }
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to extract model metadata: " + std::string(e.what()));
    }
    
    return metadata;
}

// Placeholder implementations for specific format loaders

ModelData ModelLoader::load_gguf(const std::string& file_path) {
    // Check if file exists
    if (!std::filesystem::exists(file_path)) {
        throw std::runtime_error("GGUF file does not exist: " + file_path);
    }
    
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open GGUF file: " + file_path);
    }
    
    // Read and validate header
    GGUFHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (!file.good()) {
        throw std::runtime_error("Failed to read GGUF header");
    }
    
    if (header.magic != GGUF_MAGIC) {
        throw std::runtime_error("Invalid GGUF magic number");
    }
    
    if (header.version != GGUF_VERSION) {
        throw std::runtime_error("Unsupported GGUF version: " + std::to_string(header.version));
    }
    
    ModelData model_data;
    ModelMetadata& metadata = model_data.metadata();
    
    // Initialize basic metadata
    metadata.name = std::filesystem::path(file_path).stem().string();
    metadata.architecture = "unknown";
    metadata.version = "gguf_v" + std::to_string(header.version);
    
    // Read metadata key-value pairs
    for (uint64_t i = 0; i < header.metadata_kv_count; ++i) {
        auto key = read_gguf_string(file);
        auto value_type = read_gguf_value_type(file);
        auto value = read_gguf_value(file, value_type);
        
        // Map important metadata fields
        if (key == "general.architecture") {
            metadata.architecture = value;
        } else if (key == "general.name") {
            metadata.name = value;
        } else if (key == "llama.vocab_size" || key == "gpt2.vocab_size") {
            metadata.vocab_size = static_cast<size_t>(std::stoull(value));
        } else if (key == "llama.embedding_length" || key == "gpt2.embedding_length") {
            metadata.hidden_size = static_cast<size_t>(std::stoull(value));
        } else if (key == "llama.block_count" || key == "gpt2.block_count") {
            metadata.num_layers = static_cast<size_t>(std::stoull(value));
        } else if (key == "llama.attention.head_count" || key == "gpt2.attention.head_count") {
            metadata.num_heads = static_cast<size_t>(std::stoull(value));
        } else if (key == "llama.feed_forward_length" || key == "gpt2.feed_forward_length") {
            metadata.intermediate_size = static_cast<size_t>(std::stoull(value));
        } else if (key == "llama.rope.theta") {
            metadata.rope_theta = std::stof(value);
        } else {
            // Store as extra parameter
            metadata.extra_params[key] = value;
        }
    }
    
    // Define tensor info structure for collecting tensor metadata
    struct TensorInfo {
        std::string name;
        std::vector<size_t> dims;
        core::DataType data_type;
        uint64_t offset;
        uint64_t data_size;
    };
    std::vector<TensorInfo> tensor_infos;
    
    // Read tensor information
    for (uint64_t i = 0; i < header.tensor_count; ++i) {
        auto tensor_name = read_gguf_string(file);
        
        // Read tensor dimensions
        uint32_t n_dims;
        file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
        
        std::vector<size_t> dims(n_dims);
        for (uint32_t j = 0; j < n_dims; ++j) {
            uint64_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            dims[j] = static_cast<size_t>(dim);
        }
        
        // Read tensor type
        uint32_t tensor_type;
        file.read(reinterpret_cast<char*>(&tensor_type), sizeof(tensor_type));
        
        // Read tensor offset
        uint64_t tensor_offset;
        file.read(reinterpret_cast<char*>(&tensor_offset), sizeof(tensor_offset));
        
        // Convert GGUF type to TurboInfer data type
        core::DataType data_type = convert_gguf_type_to_turboinfer(tensor_type);
        
        // GGUF uses reverse dimension order, so reverse it
        std::reverse(dims.begin(), dims.end());
        
        // Calculate tensor data size
        size_t element_count = 1;
        for (size_t dim : dims) {
            element_count *= dim;
        }
        
        size_t element_size;
        switch (data_type) {
            case core::DataType::kFloat32:
                element_size = sizeof(float);
                break;
            case core::DataType::kFloat16:
                element_size = sizeof(uint16_t); // Half precision
                break;
            default:
                element_size = sizeof(float); // Default to float32
                break;
        }
        
        uint64_t data_size = element_count * element_size;
        
        // Store tensor metadata for later data reading
        tensor_infos.push_back({
            tensor_name,
            dims,
            data_type,
            tensor_offset,
            data_size
        });
    }
    
    // Calculate the alignment padding
    // GGUF files align tensor data to 32-byte boundaries
    const size_t GGUF_ALIGNMENT = 32;
    size_t current_pos = file.tellg();
    size_t alignment_offset = (GGUF_ALIGNMENT - (current_pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
    file.seekg(current_pos + alignment_offset);
    
    // Read actual tensor data
    for (const auto& tensor_info : tensor_infos) {
        // Seek to tensor data position (relative to current aligned position)
        file.seekg(tensor_info.offset, std::ios::cur);
        
        // Create tensor with appropriate shape
        core::TensorShape tensor_shape(tensor_info.dims);
        core::Tensor tensor(tensor_shape, tensor_info.data_type);
        
        // Read tensor data directly into the tensor's memory
        void* tensor_data = tensor.data();
        file.read(reinterpret_cast<char*>(tensor_data), tensor_info.data_size);
        
        if (!file.good()) {
            throw std::runtime_error("Failed to read tensor data for: " + tensor_info.name);
        }
        
        // Store the tensor with data
        model_data.add_tensor(tensor_info.name, std::move(tensor));
    }
    
    return model_data;
}

ModelData ModelLoader::load_safetensors(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open SafeTensors file: " + file_path);
    }
    
    ModelData model_data;
    
    // Read the header size (first 8 bytes, little-endian uint64)
    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!file.good()) {
        throw std::runtime_error("Failed to read SafeTensors header size");
    }
    
    // Validate header size is reasonable (not too large)
    if (header_size > 100 * 1024 * 1024) { // 100MB max header size
        throw std::runtime_error("SafeTensors header size too large: " + std::to_string(header_size));
    }
    
    // Read the JSON header
    std::vector<char> header_buffer(header_size);
    file.read(header_buffer.data(), header_size);
    if (!file.good()) {
        throw std::runtime_error("Failed to read SafeTensors header");
    }
    
    std::string header_json(header_buffer.begin(), header_buffer.end());
    
    // Parse SafeTensors JSON format
    // Format: {"tensor_name": {"dtype": "F32", "shape": [dim1, dim2, ...], "data_offsets": [start, end]}, "__metadata__": {...}}
    
    size_t tensor_data_offset = 8 + header_size; // Start of tensor data
    
    // Simple JSON parser for SafeTensors format
    auto tensor_info = parse_safetensors_json(header_json);
    
    // Load each tensor
    for (const auto& [tensor_name, info] : tensor_info) {
        if (tensor_name == "__metadata__") {
            continue; // Skip metadata for now
        }
        
        // Convert SafeTensors dtype to TurboInfer DataType
        core::DataType data_type = convert_safetensors_dtype(info.dtype);
        
        // Create tensor shape
        core::TensorShape tensor_shape(info.shape);
        
        // Create tensor
        core::Tensor tensor(tensor_shape, data_type);
        
        // Read tensor data from file
        size_t data_size = info.data_offsets[1] - info.data_offsets[0];
        size_t expected_size = tensor.byte_size();
        
        if (data_size != expected_size) {
            throw std::runtime_error("SafeTensors tensor '" + tensor_name + "' size mismatch. Expected: " + 
                                   std::to_string(expected_size) + ", Got: " + std::to_string(data_size));
        }
        
        // Seek to tensor data position
        file.seekg(tensor_data_offset + info.data_offsets[0]);
        file.read(reinterpret_cast<char*>(tensor.data()), data_size);
        
        if (!file.good()) {
            throw std::runtime_error("Failed to read tensor data for: " + tensor_name);
        }
        
        // Add tensor to model
        model_data.add_tensor(tensor_name, std::move(tensor));
    }
    
    file.close();
    return model_data;
}

ModelData ModelLoader::load_pytorch(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open PyTorch file: " + file_path);
    }
    
    ModelData model_data;
    
    // PyTorch files are typically pickled Python objects with a specific structure
    // For a complete implementation, we would need to parse the pickle format
    // However, this requires understanding Python's pickle protocol
    
    // Check for PyTorch magic number (ZIP format signature)
    char magic[4];
    file.read(magic, 4);
    if (!file.good()) {
        throw std::runtime_error("Failed to read PyTorch file header");
    }
    
    // PyTorch files are typically ZIP archives (torch.save creates ZIP files)
    bool is_zip = (magic[0] == 'P' && magic[1] == 'K' && magic[2] == 0x03 && magic[3] == 0x04);
    
    if (!is_zip) {
        // Try old pickle format
        file.seekg(0);
        
        // Read first few bytes to check for pickle protocol
        uint8_t pickle_header[8];
        file.read(reinterpret_cast<char*>(pickle_header), 8);
        
        // Check for pickle protocol markers
        if (pickle_header[0] != 0x80) { // Pickle protocol 2+ starts with 0x80
            throw std::runtime_error("Unsupported PyTorch file format - not a valid pickle or ZIP file");
        }
        
        // For now, we'll create a minimal implementation that extracts basic information
        // A complete implementation would require a full pickle parser
        
        // Create placeholder metadata
        auto& metadata = model_data.metadata();
        metadata.name = "pytorch_model";
        metadata.architecture = "unknown";
        metadata.version = "unknown";
        metadata.vocab_size = 0;
        metadata.hidden_size = 0;
        metadata.num_layers = 0;
        metadata.num_heads = 0;
        metadata.intermediate_size = 0;
        metadata.rope_theta = 10000.0f;
        
        throw std::runtime_error("PyTorch pickle format parsing not fully implemented yet. "
                                "Use GGUF or SafeTensors format for full support.");
    }
    
    // For ZIP-based PyTorch files, we would need to:
    // 1. Extract the ZIP archive
    // 2. Read data.pkl (contains the actual tensors)
    // 3. Parse the pickle format to extract tensor data
    // 4. Convert PyTorch tensor formats to TurboInfer tensors
    
    // This is a complex task that requires implementing a pickle parser
    // For now, provide a basic structure and error message
    
    auto& metadata = model_data.metadata();
    metadata.name = "pytorch_model";
    metadata.architecture = "unknown";
    metadata.version = "unknown";
    metadata.vocab_size = 0;
    metadata.hidden_size = 0;
    metadata.num_layers = 0;
    metadata.num_heads = 0;
    metadata.intermediate_size = 0;
    metadata.rope_theta = 10000.0f;
    
    file.close();
    
    throw std::runtime_error("PyTorch format parsing not fully implemented yet. "
                            "PyTorch files require complex pickle format parsing. "
                            "Please convert your model to GGUF or SafeTensors format using "
                            "appropriate conversion tools for full support.");
}

ModelData ModelLoader::load_onnx(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open ONNX file: " + file_path);
    }
    
    ModelData model_data;
    
    // ONNX files use Protocol Buffers (protobuf) format
    // The file starts with a protobuf message containing the model graph
    
    // Read first few bytes to validate ONNX format
    char header[16];
    file.read(header, 16);
    if (!file.good()) {
        throw std::runtime_error("Failed to read ONNX file header");
    }
    
    // ONNX files typically start with protobuf wire format
    // We can check for some basic protobuf patterns
    file.seekg(0);
    
    // Read file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (file_size < 16) {
        throw std::runtime_error("ONNX file too small to be valid");
    }
    
    // For a complete ONNX implementation, we would need to:
    // 1. Parse the protobuf format (requires protobuf library or custom parser)
    // 2. Extract the model graph structure
    // 3. Read tensor data from the initializers
    // 4. Convert ONNX tensor formats to TurboInfer tensors
    // 5. Extract model metadata (input/output shapes, etc.)
    
    // Create basic metadata structure
    auto& metadata = model_data.metadata();
    metadata.name = "onnx_model";
    metadata.architecture = "unknown";
    metadata.version = "unknown";
    metadata.vocab_size = 0;
    metadata.hidden_size = 0;
    metadata.num_layers = 0;
    metadata.num_heads = 0;
    metadata.intermediate_size = 0;
    metadata.rope_theta = 10000.0f;
    
    file.close();
    
    throw std::runtime_error("ONNX format parsing not fully implemented yet. "
                            "ONNX files require protobuf parsing which needs additional dependencies. "
                            "Please convert your model to GGUF or SafeTensors format using "
                            "appropriate conversion tools for full support. "
                            "Consider using onnx2torch or similar tools to convert to a supported format.");
}

bool ModelLoader::validate_model(const ModelData& model_data, const ModelMetadata& metadata) {
    // Check if the model has any tensors
    auto tensor_names = model_data.tensor_names();
    if (tensor_names.empty()) {
        return false; // Model should have at least some tensors
    }
    
    // Basic validation checks
    bool has_essential_tensors = false;
    
    // Look for common essential tensor patterns
    for (const auto& name : tensor_names) {
        // Check for embedding or input tensors
        if (name.find("embed") != std::string::npos ||
            name.find("token") != std::string::npos ||
            name.find("wte") != std::string::npos ||
            name.find("input") != std::string::npos) {
            has_essential_tensors = true;
            break;
        }
    }
    
    // Validate tensor consistency
    for (const auto& name : tensor_names) {
        const core::Tensor* tensor = model_data.get_tensor(name);
        if (!tensor) {
            return false; // Tensor name exists but tensor is null
        }
        
        // Check if tensor has valid shape
        const auto& shape = tensor->shape();
        if (shape.dimensions().empty()) {
            return false; // Tensor should have at least one dimension
        }
        
        // Check for reasonable size limits (avoid extremely large tensors that might indicate corruption)
        size_t total_elements = 1;
        for (size_t dim : shape.dimensions()) {
            if (dim == 0) {
                return false; // No dimension should be zero
            }
            total_elements *= dim;
            if (total_elements > 1e9) { // More than 1 billion elements might indicate corruption
                return false;
            }
        }
    }
    
    // Model metadata consistency checks
    if (metadata.vocab_size == 0 || metadata.hidden_size == 0 || 
        metadata.num_layers == 0 || metadata.num_heads == 0) {
        return false; // Essential metadata should be non-zero
    }
    
    // Check if hidden_size is reasonable
    if (metadata.hidden_size > 32768) { // Very large hidden sizes might indicate issues
        return false;
    }
    
    return true; // Model passed all validation checks
}

// Utility functions

const char* format_to_string(ModelFormat format) {
    switch (format) {
        case ModelFormat::kGGUF: return "GGUF";
        case ModelFormat::kSafeTensors: return "SafeTensors";
        case ModelFormat::kPyTorch: return "PyTorch";
        case ModelFormat::kONNX: return "ONNX";
        case ModelFormat::kAuto: return "Auto";
        default: return "Unknown";
    }
}

const char* format_to_extension(ModelFormat format) {
    switch (format) {
        case ModelFormat::kGGUF: return ".gguf";
        case ModelFormat::kSafeTensors: return ".safetensors";
        case ModelFormat::kPyTorch: return ".pt";
        case ModelFormat::kONNX: return ".onnx";
        default: return "";
    }
}

bool has_valid_model_extension(const std::string& file_path) {
    std::filesystem::path path(file_path);
    std::string extension = path.extension().string();
    
    return (extension == ".gguf" || 
            extension == ".safetensors" || 
            extension == ".pt" || 
            extension == ".pth" || 
            extension == ".onnx");
}

} // namespace model
} // namespace turboinfer
