/**
 * @file model_loader.hpp
 * @brief Defines utilities for loading LLM model files in various formats.
 * @author J.J.G. Pleunes
 */

#pragma once

#include "../core/tensor.hpp"
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <cstddef>

namespace turboinfer {
namespace model {

/**
 * @enum ModelFormat
 * @brief Supported model file formats.
 */
enum class ModelFormat {
    kGGUF,      ///< GGML Universal Format
    kSafeTensors, ///< Hugging Face SafeTensors format
    kPyTorch,   ///< PyTorch .pth/.pt format
    kONNX,      ///< Open Neural Network Exchange format
    kAuto       ///< Automatic format detection
};

/**
 * @struct ModelMetadata
 * @brief Contains metadata information about a loaded model.
 */
struct ModelMetadata {
    std::string name;                           ///< Model name
    std::string architecture;                   ///< Model architecture (e.g., "llama", "gpt2")
    std::string version;                        ///< Model version
    size_t vocab_size;                          ///< Vocabulary size
    size_t hidden_size;                         ///< Hidden dimension size
    size_t num_layers;                          ///< Number of transformer layers
    size_t num_heads;                           ///< Number of attention heads
    size_t intermediate_size;                   ///< Feed-forward intermediate size
    float rope_theta;                           ///< RoPE theta parameter
    std::unordered_map<std::string, std::string> extra_params; ///< Additional parameters
};

/**
 * @class ModelData
 * @brief Container for model weights and metadata.
 */
class ModelData {
public:
    /**
     * @brief Default constructor.
     */
    ModelData() = default;

    /**
     * @brief Gets the model metadata.
     * @return Model metadata.
     */
    const ModelMetadata& metadata() const noexcept { return metadata_; }

    /**
     * @brief Gets model metadata (mutable).
     * @return Mutable model metadata.
     */
    ModelMetadata& metadata() noexcept { return metadata_; }

    /**
     * @brief Gets a tensor by name.
     * @param name Tensor name.
     * @return Tensor if found, nullptr otherwise.
     */
    const core::Tensor* get_tensor(const std::string& name) const;

    /**
     * @brief Gets a tensor by name (mutable).
     * @param name Tensor name.
     * @return Mutable tensor if found, nullptr otherwise.
     */
    core::Tensor* get_tensor(const std::string& name);

    /**
     * @brief Adds a tensor to the model.
     * @param name Tensor name.
     * @param tensor Tensor to add.
     */
    void add_tensor(const std::string& name, core::Tensor tensor);

    /**
     * @brief Gets all tensor names.
     * @return Vector of tensor names.
     */
    std::vector<std::string> tensor_names() const;

    /**
     * @brief Gets the number of tensors.
     * @return Number of tensors in the model.
     */
    size_t num_tensors() const noexcept { return tensors_.size(); }

    /**
     * @brief Checks if the model has a tensor with the given name.
     * @param name Tensor name to check.
     * @return True if tensor exists, false otherwise.
     */
    bool has_tensor(const std::string& name) const;

    /**
     * @brief Calculates the total memory usage of all tensors.
     * @return Total memory usage in bytes.
     */
    size_t total_memory_usage() const;

private:
    ModelMetadata metadata_;                                    ///< Model metadata
    std::unordered_map<std::string, core::Tensor> tensors_;    ///< Model tensors
};

/**
 * @class ModelLoader
 * @brief Loads model files in various formats and provides access to weights and metadata.
 */
class ModelLoader {
public:
    /**
     * @brief Loads a model from a file with automatic format detection.
     * @param file_path Path to the model file.
     * @return Loaded model data.
     * @throws std::runtime_error if the file cannot be read or is invalid.
     */
    static ModelData load(const std::string& file_path);

    /**
     * @brief Loads a model from a file with specified format.
     * @param file_path Path to the model file.
     * @param format Model file format.
     * @return Loaded model data.
     * @throws std::runtime_error if the file cannot be read or is invalid.
     */
    static ModelData load(const std::string& file_path, ModelFormat format);

    /**
     * @brief Detects the format of a model file.
     * @param file_path Path to the model file.
     * @return Detected model format.
     * @throws std::runtime_error if format cannot be determined.
     */
    static ModelFormat detect_format(const std::string& file_path);

    /**
     * @brief Validates that a model file exists and is readable.
     * @param file_path Path to the model file.
     * @return True if file is valid, false otherwise.
     */
    static bool validate_file(const std::string& file_path);

    /**
     * @brief Gets information about a model file without loading it.
     * @param file_path Path to the model file.
     * @return Model metadata.
     * @throws std::runtime_error if the file cannot be read.
     */
    static ModelMetadata get_model_info(const std::string& file_path);

    /**
     * @brief Validates that a loaded model is complete and consistent.
     * @param model_data The loaded model data to validate.
     * @param metadata Expected model metadata for validation.
     * @return True if model is valid, false otherwise.
     */
    static bool validate_model(const ModelData& model_data, const ModelMetadata& metadata);

private:
    /**
     * @brief Loads a GGUF format model.
     * @param file_path Path to the GGUF file.
     * @return Loaded model data.
     * @throws std::runtime_error if the file is invalid.
     */
    static ModelData load_gguf(const std::string& file_path);

    /**
     * @brief Loads a SafeTensors format model.
     * @param file_path Path to the SafeTensors file.
     * @return Loaded model data.
     * @throws std::runtime_error if the file is invalid.
     */
    static ModelData load_safetensors(const std::string& file_path);

    /**
     * @brief Loads a PyTorch format model.
     * @param file_path Path to the PyTorch file.
     * @return Loaded model data.
     * @throws std::runtime_error if the file is invalid.
     */
    static ModelData load_pytorch(const std::string& file_path);

    /**
     * @brief Loads an ONNX format model.
     * @param file_path Path to the ONNX file.
     * @return Loaded model data.
     * @throws std::runtime_error if the file is invalid.
     */
    static ModelData load_onnx(const std::string& file_path);
};

/**
 * @brief Converts a model format enum to string representation.
 * @param format Model format.
 * @return String representation of the format.
 */
const char* format_to_string(ModelFormat format);

/**
 * @brief Gets the file extension associated with a model format.
 * @param format Model format.
 * @return File extension (including the dot).
 */
const char* format_to_extension(ModelFormat format);

/**
 * @brief Checks if a file path has a valid model file extension.
 * @param file_path Path to check.
 * @return True if extension is valid, false otherwise.
 */
bool has_valid_model_extension(const std::string& file_path);

} // namespace model
} // namespace turboinfer
