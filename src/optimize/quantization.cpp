/**
 * @file quantization.cpp
 * @brief Implementation of quantization utilities.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/optimize/quantization.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

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
    throw std::runtime_error("Quantized model saving not yet implemented");
}

model::ModelData Quantizer::load_quantized_model(const std::string& model_path) {
    throw std::runtime_error("Quantized model loading not yet implemented");
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
    return 2.0f; // Placeholder
}

float Quantizer::validate_quantization_accuracy(const model::ModelData& original_model,
                                               const model::ModelData& quantized_model,
                                               const std::vector<core::Tensor>& test_inputs) {
    return 0.01f; // Placeholder: 1% error
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

} // namespace optimize
} // namespace turboinfer
