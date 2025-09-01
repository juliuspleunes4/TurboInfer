/**
 * @file quantization.hpp
 * @brief Defines quantization utilities for model optimization.
 * @author J.J.G. Pleunes
 */

#pragma once

#include "../core/tensor.hpp"
#include "../model/model_loader.hpp"
#include <memory>
#include <vector>
#include <string>
#include <cstddef>

namespace turboinfer {
namespace optimize {

/**
 * @enum QuantizationType
 * @brief Supported quantization types for model compression.
 */
enum class QuantizationType {
    kInt8,      ///< 8-bit integer quantization
    kInt4,      ///< 4-bit integer quantization
    kFloat16,   ///< 16-bit floating point
    kNone       ///< No quantization (original precision)
};

/**
 * @struct QuantizationConfig
 * @brief Configuration parameters for quantization.
 */
struct QuantizationConfig {
    QuantizationType type = QuantizationType::kInt8;    ///< Quantization type
    bool symmetric = true;                              ///< Use symmetric quantization
    bool per_channel = true;                            ///< Per-channel vs per-tensor quantization
    float calibration_ratio = 0.1f;                     ///< Fraction of data for calibration
    std::string calibration_dataset;                    ///< Path to calibration dataset
};

/**
 * @struct QuantizationInfo
 * @brief Information about quantized tensors.
 */
struct QuantizationInfo {
    QuantizationType type;                  ///< Quantization type used
    std::vector<float> scales;              ///< Quantization scales
    std::vector<float> zero_points;         ///< Zero point offsets
    size_t original_size_bytes;             ///< Original tensor size in bytes
    size_t quantized_size_bytes;            ///< Quantized tensor size in bytes
    float compression_ratio;                ///< Compression ratio (original/quantized)
};

/**
 * @class Quantizer
 * @brief Handles quantization of model weights and activations.
 * 
 * The Quantizer class provides methods to reduce model size and memory usage
 * through various quantization techniques while maintaining acceptable accuracy.
 */
class Quantizer {
public:
    /**
     * @brief Constructs a quantizer with the specified configuration.
     * @param config Quantization configuration.
     */
    explicit Quantizer(const QuantizationConfig& config = QuantizationConfig{});

    /**
     * @brief Destructor.
     */
    ~Quantizer();

    /**
     * @brief Gets the quantization configuration.
     * @return Quantization configuration.
     */
    const QuantizationConfig& config() const noexcept { return config_; }

    /**
     * @brief Updates the quantization configuration.
     * @param config New configuration.
     */
    void set_config(const QuantizationConfig& config);

    /**
     * @brief Quantizes a single tensor.
     * @param input Input tensor to quantize.
     * @return Quantized tensor.
     * @throws std::runtime_error if quantization fails.
     */
    core::Tensor quantize_tensor(const core::Tensor& input);

    /**
     * @brief Dequantizes a tensor back to its original precision.
     * @param quantized Quantized tensor.
     * @param info Quantization information.
     * @return Dequantized tensor.
     * @throws std::runtime_error if dequantization fails.
     */
    core::Tensor dequantize_tensor(const core::Tensor& quantized, const QuantizationInfo& info);

    /**
     * @brief Quantizes an entire model.
     * @param model_data Input model data.
     * @return Quantized model data.
     * @throws std::runtime_error if quantization fails.
     */
    model::ModelData quantize_model(const model::ModelData& model_data);

    /**
     * @brief Saves a quantized model to file.
     * @param quantized_model Quantized model data.
     * @param output_path Output file path.
     * @throws std::runtime_error if saving fails.
     */
    void save_quantized_model(const model::ModelData& quantized_model, 
                             const std::string& output_path);

    /**
     * @brief Loads a quantized model from file.
     * @param model_path Path to quantized model file.
     * @return Quantized model data.
     * @throws std::runtime_error if loading fails.
     */
    static model::ModelData load_quantized_model(const std::string& model_path);

    /**
     * @brief Calculates quantization statistics for a tensor.
     * @param input Input tensor.
     * @return Quantization information.
     */
    QuantizationInfo calculate_quantization_info(const core::Tensor& input);

    /**
     * @brief Estimates the memory savings from quantization.
     * @param model_data Original model data.
     * @return Estimated compression ratio.
     */
    float estimate_compression_ratio(const model::ModelData& model_data);

    /**
     * @brief Validates quantization accuracy by comparing outputs.
     * @param original_model Original model.
     * @param quantized_model Quantized model.
     * @param test_inputs Test input tensors.
     * @return Average relative error between outputs.
     */
    float validate_quantization_accuracy(const model::ModelData& original_model,
                                        const model::ModelData& quantized_model,
                                        const std::vector<core::Tensor>& test_inputs);

private:
    QuantizationConfig config_;                         ///< Quantization configuration
    std::unique_ptr<class QuantizerImpl> impl_;         ///< Implementation details

    /**
     * @brief Initializes the quantizer.
     */
    void initialize();

    /**
     * @brief Determines quantization parameters for a tensor.
     * @param input Input tensor.
     * @return Quantization information.
     */
    QuantizationInfo determine_quantization_params(const core::Tensor& input);

    /**
     * @brief Applies quantization to tensor data.
     * @param input Input tensor data.
     * @param info Quantization parameters.
     * @return Quantized data.
     */
    std::vector<uint8_t> apply_quantization(const core::Tensor& input, 
                                           const QuantizationInfo& info);
    
    /**
     * @brief Helper function to write strings to binary files.
     * @param file Output file stream.
     * @param str String to write.
     */
    static void write_string(std::ofstream& file, const std::string& str);
    
    /**
     * @brief Helper function to read strings from binary files.
     * @param file Input file stream.
     * @return Read string.
     */
    static std::string read_string(std::ifstream& file);
    
    /**
     * @brief Calculate quantization info for a saved tensor.
     * @param tensor Quantized tensor.
     * @return Quantization information.
     */
    QuantizationInfo calculate_quantization_info_for_saved_tensor(const core::Tensor& tensor);
};

/**
 * @brief Converts quantization type to string representation.
 * @param type Quantization type.
 * @return String representation.
 */
const char* quantization_type_to_string(QuantizationType type);

/**
 * @brief Gets the number of bits for a quantization type.
 * @param type Quantization type.
 * @return Number of bits per element.
 */
size_t get_quantization_bits(QuantizationType type);

/**
 * @brief Calculates the theoretical compression ratio for a quantization type.
 * @param from_type Original data type.
 * @param to_type Target quantization type.
 * @return Theoretical compression ratio.
 */
float calculate_theoretical_compression(core::DataType from_type, QuantizationType to_type);

/**
 * @brief Convenience function to quantize a model from file.
 * @param input_path Path to input model file.
 * @param output_path Path to output quantized model file.
 * @param config Quantization configuration.
 * @throws std::runtime_error if quantization fails.
 */
void quantize_model_file(const std::string& input_path,
                        const std::string& output_path,
                        const QuantizationConfig& config = QuantizationConfig{});

// Helper functions for quantization algorithms (internal use)

/**
 * @brief Quantizes float array to int8.
 * @param input Input float array.
 * @param output Output int8 array.
 * @param count Number of elements.
 * @param info Quantization parameters.
 */
void quantize_to_int8(const float* input, int8_t* output, size_t count, const QuantizationInfo& info);

/**
 * @brief Quantizes float array to int4 (stored as int32).
 * @param input Input float array.
 * @param output Output int32 array (stores int4 values).
 * @param count Number of elements.
 * @param info Quantization parameters.
 */
void quantize_to_int4(const float* input, int32_t* output, size_t count, const QuantizationInfo& info);

/**
 * @brief Dequantizes int8 array to float.
 * @param input Input int8 array.
 * @param output Output float array.
 * @param count Number of elements.
 * @param info Quantization parameters.
 */
void dequantize_from_int8(const int8_t* input, float* output, size_t count, const QuantizationInfo& info);

/**
 * @brief Dequantizes int4 array (stored as int32) to float.
 * @param input Input int32 array (contains int4 values).
 * @param output Output float array.
 * @param count Number of elements.
 * @param info Quantization parameters.
 */
void dequantize_from_int4(const int32_t* input, float* output, size_t count, const QuantizationInfo& info);

} // namespace optimize
} // namespace turboinfer
