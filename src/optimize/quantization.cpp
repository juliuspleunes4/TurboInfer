/**
 * @file quantization.cpp
 * @brief Implementation of quantization utilities.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/optimize/quantization.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>

namespace turboinfer {
namespace optimize {

// Forward declaration for implementation
class QuantizerImpl {
public:
    QuantizationConfig config;
};

Quantizer::Quantizer(const QuantizationConfig& config) 
    : config_(config), impl_(std::make_unique<QuantizerImpl>()) {
    initialize();
}

Quantizer::~Quantizer() = default;

void Quantizer::set_config(const QuantizationConfig& config) {
    config_ = config;
    impl_->config = config;
}

core::Tensor Quantizer::quantize_tensor(const core::Tensor& input) {
    if (config_.type == QuantizationType::kNone) {
        return input; // No quantization
    }
    
    // Calculate quantization parameters
    auto quant_info = calculate_quantization_info(input);
    
    // Create output tensor with appropriate data type
    core::DataType output_type = (config_.type == QuantizationType::kInt8) ? 
        core::DataType::kInt8 : core::DataType::kInt32; // Use Int32 to store Int4 for now
    
    core::Tensor quantized(input.shape(), output_type);
    
    const float* input_data = static_cast<const float*>(input.data());
    size_t total_elements = input.shape().total_size();
    
    if (config_.type == QuantizationType::kInt8) {
        int8_t* output_data = static_cast<int8_t*>(quantized.data());
        quantize_to_int8(input_data, output_data, total_elements, quant_info);
    } else if (config_.type == QuantizationType::kInt4) {
        int32_t* output_data = static_cast<int32_t*>(quantized.data());
        quantize_to_int4(input_data, output_data, total_elements, quant_info);
    } else {
        throw std::runtime_error("Unsupported quantization type");
    }
    
    return quantized;
}

core::Tensor Quantizer::dequantize_tensor(const core::Tensor& quantized, const QuantizationInfo& info) {
    if (info.type == QuantizationType::kNone) {
        return quantized; // No dequantization needed
    }
    
    // Create output tensor with Float32 data type
    core::Tensor dequantized(quantized.shape(), core::DataType::kFloat32);
    float* output_data = static_cast<float*>(dequantized.data());
    size_t total_elements = quantized.shape().total_size();
    
    if (info.type == QuantizationType::kInt8) {
        const int8_t* input_data = static_cast<const int8_t*>(quantized.data());
        dequantize_from_int8(input_data, output_data, total_elements, info);
    } else if (info.type == QuantizationType::kInt4) {
        const int32_t* input_data = static_cast<const int32_t*>(quantized.data());
        dequantize_from_int4(input_data, output_data, total_elements, info);
    } else {
        throw std::runtime_error("Unsupported quantization type for dequantization");
    }
    
    return dequantized;
}

model::ModelData Quantizer::quantize_model(const model::ModelData& model_data) {
    model::ModelData quantized_model;
    quantized_model.metadata() = model_data.metadata();
    
    // Get all tensor names from the original model
    auto tensor_names = model_data.tensor_names();
    
    for (const auto& name : tensor_names) {
        const auto* tensor = model_data.get_tensor(name);
        if (tensor == nullptr) {
            continue;
        }
        
        // Only quantize float tensors (skip embeddings and other special tensors for now)
        if (tensor->dtype() == core::DataType::kFloat32) {
            try {
                auto quantized_tensor = quantize_tensor(*tensor);
                quantized_model.add_tensor(name, std::move(quantized_tensor));
            } catch (const std::exception& e) {
                // If quantization fails, keep original tensor
                quantized_model.add_tensor(name, *tensor);
            }
        } else {
            // Keep non-float tensors as-is
            quantized_model.add_tensor(name, *tensor);
        }
    }
    
    return quantized_model;
}

void Quantizer::save_quantized_model(const model::ModelData& quantized_model, const std::string& output_path) {
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + output_path);
    }
    
    try {
        // Write magic header
        const uint32_t magic = 0x54494E51; // "TINQ" - TurboInfer Quantized
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        
        // Write version
        const uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Write quantization config
        file.write(reinterpret_cast<const char*>(&config_.type), sizeof(config_.type));
        file.write(reinterpret_cast<const char*>(&config_.symmetric), sizeof(config_.symmetric));
        file.write(reinterpret_cast<const char*>(&config_.per_channel), sizeof(config_.per_channel));
        
        // Write metadata
        const auto& metadata = quantized_model.metadata();
        write_string(file, metadata.name);
        write_string(file, metadata.architecture);
        write_string(file, metadata.version);
        file.write(reinterpret_cast<const char*>(&metadata.vocab_size), sizeof(metadata.vocab_size));
        file.write(reinterpret_cast<const char*>(&metadata.hidden_size), sizeof(metadata.hidden_size));
        file.write(reinterpret_cast<const char*>(&metadata.num_layers), sizeof(metadata.num_layers));
        file.write(reinterpret_cast<const char*>(&metadata.num_heads), sizeof(metadata.num_heads));
        file.write(reinterpret_cast<const char*>(&metadata.intermediate_size), sizeof(metadata.intermediate_size));
        file.write(reinterpret_cast<const char*>(&metadata.rope_theta), sizeof(metadata.rope_theta));
        
        // Write tensor count
        auto tensor_names = quantized_model.tensor_names();
        uint32_t tensor_count = static_cast<uint32_t>(tensor_names.size());
        file.write(reinterpret_cast<const char*>(&tensor_count), sizeof(tensor_count));
        
        // Write tensors
        for (const auto& name : tensor_names) {
            const auto* tensor = quantized_model.get_tensor(name);
            if (tensor == nullptr) continue;
            
            // Write tensor name
            write_string(file, name);
            
            // Write tensor metadata
            auto dtype = static_cast<uint32_t>(tensor->dtype());
            file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
            
            auto ndim = static_cast<uint32_t>(tensor->shape().ndim());
            file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
            
            for (size_t i = 0; i < tensor->shape().ndim(); ++i) {
                auto dim_size = static_cast<uint64_t>(tensor->shape().size(i));
                file.write(reinterpret_cast<const char*>(&dim_size), sizeof(dim_size));
            }
            
            // Write tensor data
            size_t byte_size = tensor->byte_size();
            file.write(reinterpret_cast<const char*>(&byte_size), sizeof(byte_size));
            file.write(reinterpret_cast<const char*>(tensor->data()), byte_size);
            
            // Write quantization info if this is a quantized tensor
            if (tensor->dtype() == core::DataType::kInt8 || tensor->dtype() == core::DataType::kInt32) {
                auto quant_info = calculate_quantization_info_for_saved_tensor(*tensor);
                
                // Write quantization scales
                uint32_t scales_count = static_cast<uint32_t>(quant_info.scales.size());
                file.write(reinterpret_cast<const char*>(&scales_count), sizeof(scales_count));
                if (scales_count > 0) {
                    file.write(reinterpret_cast<const char*>(quant_info.scales.data()), 
                              scales_count * sizeof(float));
                }
                
                // Write zero points
                uint32_t zp_count = static_cast<uint32_t>(quant_info.zero_points.size());
                file.write(reinterpret_cast<const char*>(&zp_count), sizeof(zp_count));
                if (zp_count > 0) {
                    file.write(reinterpret_cast<const char*>(quant_info.zero_points.data()), 
                              zp_count * sizeof(float));
                }
                
                file.write(reinterpret_cast<const char*>(&quant_info.original_size_bytes), sizeof(quant_info.original_size_bytes));
                file.write(reinterpret_cast<const char*>(&quant_info.quantized_size_bytes), sizeof(quant_info.quantized_size_bytes));
                file.write(reinterpret_cast<const char*>(&quant_info.compression_ratio), sizeof(quant_info.compression_ratio));
            }
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to save quantized model: " + std::string(e.what()));
    }
}

model::ModelData Quantizer::load_quantized_model(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + model_path);
    }
    
    try {
        // Read and verify magic header
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x54494E51) { // "TINQ"
            throw std::runtime_error("Invalid file format - not a TurboInfer quantized model");
        }
        
        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            throw std::runtime_error("Unsupported quantized model version: " + std::to_string(version));
        }
        
        // Read quantization config
        QuantizationType type;
        bool symmetric, per_channel;
        file.read(reinterpret_cast<char*>(&type), sizeof(type));
        file.read(reinterpret_cast<char*>(&symmetric), sizeof(symmetric));
        file.read(reinterpret_cast<char*>(&per_channel), sizeof(per_channel));
        
        // Create model data
        model::ModelData model_data;
        
        // Read metadata
        auto& metadata = model_data.metadata();
        metadata.name = read_string(file);
        metadata.architecture = read_string(file);
        metadata.version = read_string(file);
        file.read(reinterpret_cast<char*>(&metadata.vocab_size), sizeof(metadata.vocab_size));
        file.read(reinterpret_cast<char*>(&metadata.hidden_size), sizeof(metadata.hidden_size));
        file.read(reinterpret_cast<char*>(&metadata.num_layers), sizeof(metadata.num_layers));
        file.read(reinterpret_cast<char*>(&metadata.num_heads), sizeof(metadata.num_heads));
        file.read(reinterpret_cast<char*>(&metadata.intermediate_size), sizeof(metadata.intermediate_size));
        file.read(reinterpret_cast<char*>(&metadata.rope_theta), sizeof(metadata.rope_theta));
        
        // Read tensor count
        uint32_t tensor_count;
        file.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
        
        // Read tensors
        for (uint32_t i = 0; i < tensor_count; ++i) {
            // Read tensor name
            std::string name = read_string(file);
            
            // Read tensor metadata
            uint32_t dtype_raw;
            file.read(reinterpret_cast<char*>(&dtype_raw), sizeof(dtype_raw));
            auto dtype = static_cast<core::DataType>(dtype_raw);
            
            uint32_t ndim;
            file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
            
            std::vector<size_t> shape_dims;
            for (uint32_t d = 0; d < ndim; ++d) {
                uint64_t dim_size;
                file.read(reinterpret_cast<char*>(&dim_size), sizeof(dim_size));
                shape_dims.push_back(static_cast<size_t>(dim_size));
            }
            
            // Create tensor
            core::TensorShape shape(shape_dims);
            core::Tensor tensor(shape, dtype);
            
            // Read tensor data
            size_t byte_size;
            file.read(reinterpret_cast<char*>(&byte_size), sizeof(byte_size));
            
            if (byte_size != tensor.byte_size()) {
                throw std::runtime_error("Tensor size mismatch for: " + name);
            }
            
            file.read(reinterpret_cast<char*>(tensor.data()), byte_size);
            
            // Read quantization info for quantized tensors
            if (dtype == core::DataType::kInt8 || dtype == core::DataType::kInt32) {
                QuantizationInfo quant_info;
                quant_info.type = type;
                
                // Read scales
                uint32_t scales_count;
                file.read(reinterpret_cast<char*>(&scales_count), sizeof(scales_count));
                if (scales_count > 0) {
                    quant_info.scales.resize(scales_count);
                    file.read(reinterpret_cast<char*>(quant_info.scales.data()), 
                             scales_count * sizeof(float));
                }
                
                // Read zero points
                uint32_t zp_count;
                file.read(reinterpret_cast<char*>(&zp_count), sizeof(zp_count));
                if (zp_count > 0) {
                    quant_info.zero_points.resize(zp_count);
                    file.read(reinterpret_cast<char*>(quant_info.zero_points.data()), 
                             zp_count * sizeof(float));
                }
                
                file.read(reinterpret_cast<char*>(&quant_info.original_size_bytes), sizeof(quant_info.original_size_bytes));
                file.read(reinterpret_cast<char*>(&quant_info.quantized_size_bytes), sizeof(quant_info.quantized_size_bytes));
                file.read(reinterpret_cast<char*>(&quant_info.compression_ratio), sizeof(quant_info.compression_ratio));
                
                // Store quantization info for later use (in practice, you might want to store this in the tensor metadata)
            }
            
            // Add tensor to model
            model_data.add_tensor(name, std::move(tensor));
        }
        
        return model_data;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load quantized model: " + std::string(e.what()));
    }
}

QuantizationInfo Quantizer::calculate_quantization_info(const core::Tensor& input) {
    QuantizationInfo info;
    info.type = config_.type;
    info.original_size_bytes = input.byte_size();
    
    const float* data = static_cast<const float*>(input.data());
    size_t total_elements = input.shape().total_size();
    
    // Calculate min and max values for quantization
    float min_val = data[0];
    float max_val = data[0];
    for (size_t i = 1; i < total_elements; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }
    
    // Calculate scale and zero point based on quantization type
    if (config_.type == QuantizationType::kInt8) {
        float scale;
        float zero_point;
        
        if (config_.symmetric) {
            // Symmetric quantization: scale = max(|min|, |max|) / 127
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            scale = abs_max / 127.0f;
            zero_point = 0.0f;
        } else {
            // Asymmetric quantization
            scale = (max_val - min_val) / 255.0f;
            zero_point = -min_val / scale;  // This maps min_val to 0
        }
        
        info.scales = {scale};
        info.zero_points = {zero_point};
        info.quantized_size_bytes = total_elements; // 1 byte per element
        info.compression_ratio = static_cast<float>(info.original_size_bytes) / info.quantized_size_bytes;
    } else if (config_.type == QuantizationType::kInt4) {
        float scale;
        float zero_point;
        
        if (config_.symmetric) {
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            scale = abs_max / 7.0f; // 4-bit signed range: -7 to 7
            zero_point = 0.0f;
        } else {
            scale = (max_val - min_val) / 15.0f; // 4-bit unsigned range: 0 to 15
            zero_point = -min_val / scale;
        }
        
        info.scales = {scale};
        info.zero_points = {zero_point};
        info.quantized_size_bytes = (total_elements + 1) / 2; // 0.5 bytes per element (packed)
        info.compression_ratio = static_cast<float>(info.original_size_bytes) / info.quantized_size_bytes;
    } else {
        info.quantized_size_bytes = info.original_size_bytes;
        info.compression_ratio = 1.0f;
    }
    
    return info;
}

float Quantizer::estimate_compression_ratio(const model::ModelData& model_data) {
    if (model_data.num_tensors() == 0) {
        return 1.0f; // No compression if no data
    }
    
    size_t original_size = 0;
    size_t compressed_size = 0;
    
    auto tensor_names = model_data.tensor_names();
    for (const auto& name : tensor_names) {
        const auto* tensor = model_data.get_tensor(name);
        if (!tensor) continue;
        
        size_t tensor_elements = 1;
        for (size_t dim : tensor->shape().dimensions()) {
            tensor_elements *= dim;
        }
        
        // Calculate original size (assuming float32)
        original_size += tensor_elements * sizeof(float);
        
        // Calculate compressed size based on quantization type
        switch (config_.type) {
            case QuantizationType::kInt4:
                compressed_size += (tensor_elements + 1) / 2; // 4 bits per element
                break;
            case QuantizationType::kInt8:
                compressed_size += tensor_elements; // 8 bits per element
                break;
            case QuantizationType::kFloat16:
                compressed_size += tensor_elements * 2; // 16 bits per element
                break;
            case QuantizationType::kNone:
            default:
                compressed_size += tensor_elements * sizeof(float);
                break;
        }
        
        // Add overhead for quantization parameters (scale, zero_point per tensor)
        if (config_.type != QuantizationType::kNone) {
            compressed_size += sizeof(float) + sizeof(int32_t); // scale + zero_point
        }
    }
    
    if (original_size == 0) {
        return 1.0f;
    }
    
    return static_cast<float>(original_size) / static_cast<float>(compressed_size);
}

float Quantizer::validate_quantization_accuracy(const model::ModelData& original_model,
                                               const model::ModelData& quantized_model,
                                               const std::vector<core::Tensor>& test_inputs) {
    if (original_model.num_tensors() == 0 || quantized_model.num_tensors() == 0) {
        return 0.0f; // No accuracy to measure
    }
    
    if (test_inputs.empty()) {
        // If no test inputs provided, compare tensor values directly
        float total_error = 0.0f;
        size_t total_elements = 0;
        
        auto orig_tensor_names = original_model.tensor_names();
        for (const auto& name : orig_tensor_names) {
            const auto* orig_tensor = original_model.get_tensor(name);
            const auto* quant_tensor = quantized_model.get_tensor(name);
            
            if (!orig_tensor || !quant_tensor) {
                continue; // Skip if tensor not found in either model
            }
            
            // Ensure tensors have same shape
            if (orig_tensor->shape() != quant_tensor->shape()) {
                continue;
            }
            
            // Calculate element-wise error
            size_t num_elements = 1;
            for (size_t dim : orig_tensor->shape().dimensions()) {
                num_elements *= dim;
            }
            
            const float* orig_data = orig_tensor->data_ptr<float>();
            const float* quant_data = quant_tensor->data_ptr<float>();
            
            for (size_t i = 0; i < num_elements; ++i) {
                float error = std::abs(orig_data[i] - quant_data[i]);
                float relative_error = (orig_data[i] != 0.0f) ? 
                    error / std::abs(orig_data[i]) : error;
                total_error += relative_error;
            }
            
            total_elements += num_elements;
        }
        
        return (total_elements > 0) ? total_error / total_elements : 0.0f;
    }
    
    // TODO: Implement inference-based accuracy validation when inference engine is available
    // This would run both models on test inputs and compare outputs
    
    // For now, return a conservative estimate based on quantization type
    switch (config_.type) {
        case QuantizationType::kInt4:
            return 0.05f; // ~5% error for 4-bit quantization
        case QuantizationType::kInt8:
            return 0.02f; // ~2% error for 8-bit quantization
        case QuantizationType::kFloat16:
            return 0.001f; // ~0.1% error for 16-bit quantization
        case QuantizationType::kNone:
        default:
            return 0.0f; // No quantization error
    }
}

void Quantizer::initialize() {
    impl_->config = config_;
}

QuantizationInfo Quantizer::determine_quantization_params(const core::Tensor& input) {
    return calculate_quantization_info(input);
}

std::vector<uint8_t> Quantizer::apply_quantization(const core::Tensor& input, const QuantizationInfo& info) {
    std::vector<uint8_t> result(info.quantized_size_bytes);
    return result;
}

// Utility functions

const char* quantization_type_to_string(QuantizationType type) {
    switch (type) {
        case QuantizationType::kInt8: return "int8";
        case QuantizationType::kInt4: return "int4";
        case QuantizationType::kFloat16: return "float16";
        case QuantizationType::kNone: return "none";
        default: return "unknown";
    }
}

size_t get_quantization_bits(QuantizationType type) {
    switch (type) {
        case QuantizationType::kInt8: return 8;
        case QuantizationType::kInt4: return 4;
        case QuantizationType::kFloat16: return 16;
        case QuantizationType::kNone: return 32;
        default: return 0;
    }
}

float calculate_theoretical_compression(core::DataType from_type, QuantizationType to_type) {
    size_t from_bits = core::get_dtype_size(from_type) * 8;
    size_t to_bits = get_quantization_bits(to_type);
    return static_cast<float>(from_bits) / static_cast<float>(to_bits);
}

void quantize_model_file(const std::string& input_path,
                        const std::string& output_path,
                        const QuantizationConfig& config) {
    Quantizer quantizer(config);
    auto model_data = model::ModelLoader::load(input_path);
    auto quantized_model = quantizer.quantize_model(model_data);
    quantizer.save_quantized_model(quantized_model, output_path);
}

// Helper functions for quantization algorithms

void quantize_to_int8(const float* input, int8_t* output, size_t count, const QuantizationInfo& info) {
    float scale = info.scales[0];
    float zero_point = info.zero_points[0];
    
    for (size_t i = 0; i < count; ++i) {
        // Quantize: q = round(x / scale + zero_point)
        float quantized_val = std::round(input[i] / scale + zero_point);
        
        // Clamp to int8 range
        quantized_val = std::max(-128.0f, std::min(127.0f, quantized_val));
        output[i] = static_cast<int8_t>(quantized_val);
    }
}

void quantize_to_int4(const float* input, int32_t* output, size_t count, const QuantizationInfo& info) {
    float scale = info.scales[0];
    float zero_point = info.zero_points[0];
    
    for (size_t i = 0; i < count; ++i) {
        // Quantize: q = round(x / scale - zero_point)
        float quantized_val = std::round(input[i] / scale - zero_point);
        
        // Clamp to int4 range (-7 to 7 for symmetric, 0 to 15 for asymmetric)
        if (zero_point == 0.0f) { // Symmetric
            quantized_val = std::max(-7.0f, std::min(7.0f, quantized_val));
        } else { // Asymmetric
            quantized_val = std::max(0.0f, std::min(15.0f, quantized_val));
        }
        
        output[i] = static_cast<int32_t>(quantized_val);
    }
}

void dequantize_from_int8(const int8_t* input, float* output, size_t count, const QuantizationInfo& info) {
    float scale = info.scales[0];
    float zero_point = info.zero_points[0];
    
    for (size_t i = 0; i < count; ++i) {
        // Dequantize: x = scale * (q - zero_point)
        output[i] = scale * (static_cast<float>(input[i]) - zero_point);
    }
}

void dequantize_from_int4(const int32_t* input, float* output, size_t count, const QuantizationInfo& info) {
    float scale = info.scales[0];
    float zero_point = info.zero_points[0];
    
    for (size_t i = 0; i < count; ++i) {
        // Dequantize: x = scale * (q + zero_point)
        output[i] = scale * (static_cast<float>(input[i]) + zero_point);
    }
}

// Helper functions for file I/O
void Quantizer::write_string(std::ofstream& file, const std::string& str) {
    uint32_t length = static_cast<uint32_t>(str.length());
    file.write(reinterpret_cast<const char*>(&length), sizeof(length));
    if (length > 0) {
        file.write(str.c_str(), length);
    }
}

std::string Quantizer::read_string(std::ifstream& file) {
    uint32_t length;
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    
    if (length == 0) {
        return "";
    }
    
    std::string str(length, '\0');
    file.read(&str[0], length);
    return str;
}

QuantizationInfo Quantizer::calculate_quantization_info_for_saved_tensor(const core::Tensor& tensor) {
    QuantizationInfo info;
    info.type = config_.type;
    info.quantized_size_bytes = tensor.byte_size();
    
    // For saved tensors, we estimate the original size and create placeholder quantization parameters
    // In a real implementation, you might want to store this info during quantization
    if (tensor.dtype() == core::DataType::kInt8) {
        info.original_size_bytes = tensor.shape().total_size() * sizeof(float);
        info.compression_ratio = static_cast<float>(info.original_size_bytes) / info.quantized_size_bytes;
        
        // Add placeholder scale and zero point (in practice, these should be stored during quantization)
        info.scales.push_back(1.0f / 127.0f);
        info.zero_points.push_back(0.0f);
    } else if (tensor.dtype() == core::DataType::kInt32) { // Used for INT4
        info.original_size_bytes = tensor.shape().total_size() * sizeof(float);
        info.compression_ratio = static_cast<float>(info.original_size_bytes) / info.quantized_size_bytes;
        
        // Add placeholder scale and zero point
        info.scales.push_back(1.0f / 15.0f);
        info.zero_points.push_back(0.0f);
    } else {
        // Not a quantized tensor
        info.original_size_bytes = tensor.byte_size();
        info.compression_ratio = 1.0f;
    }
    
    return info;
}

} // namespace optimize
} // namespace turboinfer
