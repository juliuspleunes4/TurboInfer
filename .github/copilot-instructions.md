# Copilot Instructions for TurboInfer

## Project Overview

**TurboInfer** is an open-source C++ library designed to accelerate inference for large language models (LLMs) in production environments. It enables low-latency, high-throughput execution of transformer-based models (e.g., GPT-2, LLaMA) on diverse hardware, including CPUs, GPUs, and potentially edge devices. The library handles model loading, tokenization, quantization (4-bit/8-bit), batched inference, and parallel processing, optimized for performance and scalability. It aims to be a lightweight, embeddable alternative to Python-based frameworks, targeting enterprise and research use cases. The project is licensed under the **Apache 2.0 License** to encourage open-source contributions.

The codebase must adhere to professional standards, prioritizing clean structure, exceptional documentation, and maintainability to ensure it is accessible to contributors and users in the AI/ML community.

## Development Environment

- **IDE**: Use Visual Studio Code with extensions for C++ (e.g., Microsoft C/C++), CMake Tools, and Git integration.
- **AI Assistance**: Use VSCode Copilot powered by Claude to assist with code generation, refactoring, and suggestions. Ensure Claude follows these guidelines strictly.
- **Build System**: Use CMake for cross-platform builds.
- **Compiler**: Target C++20 standard, compatible with GCC, Clang, and MSVC.
- **Dependencies**: Minimal external libraries (e.g., Eigen for linear algebra, Cereal for serialization, GoogleTest for testing). Avoid heavy dependencies like PyTorch.
- **Version Control**: Git, hosted on GitHub, with a clear branching strategy (e.g., `main`, `dev`, feature branches).

## Code Organization (example)

Organize the project using a modular, hierarchical directory structure to promote clarity and scalability. Below is the recommended structure:

```
TurboInfer/
├── include/                    # Public headers
│   └── turboinfer/
│       ├── core/           # Core tensor and inference logic
│       ├── model/          # Model loading and parsing
│       ├── optimize/       # Quantization and optimization utilities
│       └── util/           # General utilities (logging, profiling)
├── src/                        # Implementation files
│   ├── core/
│   ├── model/
│   ├── optimize/
│   └── util/
├── tests/                      # Unit and integration tests
├── examples/                   # Sample applications (e.g., CLI inference)
├── benchmarks/                 # Performance benchmarking scripts
├── docs/                       # Documentation (API, guides)
├── cmake/                      # CMake modules for dependencies
├── CMakeLists.txt              # Root CMake configuration
├── LICENSE                     # Apache 2.0 License
├── README.md                   # Project overview and setup guide
└── copilot-instructions.md     # This file
```

- **Headers in** `include/turboinfer/`: Expose public APIs with clear interfaces.
- **Source files in** `src/`: Implementation details, one file per class or major function.
- **Tests in** `tests/`: Unit tests for each module using GoogleTest.
- **Examples in** `examples/`: Minimal, well-documented demos (e.g., running inference on a small model).
- **Benchmarks in** `benchmarks/`: Scripts to measure tokens/sec, memory usage, etc.

## Coding Standards

Adhere to modern C++20 practices and professional conventions to ensure maintainability and performance.

### General Guidelines

- **Style**: Follow Google C++ Style Guide for consistency (e.g., snake_case for variables, CamelCase for classes).
- **Modularity**: Use C++20 modules if supported; otherwise, header guards (`#pragma once`).
- **Safety**: Use RAII for resource management, `std::unique_ptr`/`std::shared_ptr` for memory, and avoid raw pointers where possible.
- **Performance**: Leverage SIMD (e.g., AVX-512 via intrinsics), multithreading (std::thread or Intel TBB), and cache-friendly data structures.
- **Error Handling**: Use exceptions for fatal errors, `std::expected` for recoverable errors. Avoid error codes unless interfacing with C APIs.
- **Portability**: Ensure cross-platform compatibility (Linux, Windows, macOS). Use CMake conditionals for platform-specific code.

### File Structure

- **Header Files**:
  - Include only necessary headers.
  - Use forward declarations to minimize dependencies.
  - Group includes: standard library, third-party, project-specific.
- **Source Files**:
  - One major class or function per file.
  - Group related functions into namespaces (e.g., `turboinfer::core`).
- **Naming**:
  - Files: `snake_case.cpp` and `snake_case.hpp`.
  - Classes: `CamelCase` (e.g., `TensorEngine`).
  - Functions/Variables: `snake_case` (e.g., `compute_attention`).

## Documentation Standards

Exceptional documentation is critical for an open-source project. Every file, class, function, and major variable must be documented clearly and concisely.

### General Documentation Rules

- **Purpose**: Document **what** the code does and **why** it exists, not **how** it was changed. Avoid references to code modifications (e.g., do not write `// Now set to 5` or `// Changed to use vector`).
- **Clarity**: Use simple, precise language suitable for developers unfamiliar with the project.
- **Format**: Use Doxygen-compatible comments for automatic API documentation generation.
- **Location**: Place documentation in header files for public APIs and in source files for implementation details.

### Specific Guidelines

- **File-Level Comments**:
  - At the top of each file, include a brief description of its purpose and contents.
  - Example:

    ```cpp
    /**
     * @file tensor_engine.hpp
     * @brief Defines the TensorEngine class for managing tensor operations in TurboInfer.
     * @author J.J.G. Pleunes
     */
    ```
- **Class Comments**:
  - Describe the class’s purpose, key responsibilities, and usage.
  - Example:

    ```cpp
    /**
     * @class TensorEngine
     * @brief Manages tensor operations (e.g., matrix multiplication, attention) for LLM inference.
     * Provides CPU and GPU backends with automatic fallback to CPU if GPU is unavailable.
     */
    class TensorEngine { ... };
    ```
- **Function Comments**:
  - Document parameters, return values, exceptions, and the function’s purpose.
  - Example:

    ```cpp
    /**
     * @brief Computes self-attention for a given input tensor.
     * @param input Tensor containing input data (batch_size, seq_len, hidden_size).
     * @param weights Tensor containing attention weights.
     * @return Tensor containing attention output.
     * @throws std::runtime_error if input dimensions are invalid.
     */
    Tensor compute_attention(const Tensor& input, const Tensor& weights);
    ```
- **Variable Comments**:
  - Document non-obvious variables or constants, especially in headers.
  - Example:

    ```cpp
    // Maximum batch size supported for inference.
    static constexpr size_t kMaxBatchSize = 64;
    ```
- **Avoid Change-Based Comments**:
  - **Incorrect**: `int batch_size = 32; // Changed from 16 to support larger models.`
  - **Correct**: `int batch_size = 32; // Default batch size for inference, adjustable via config.`

### Copilot-Specific Instructions for Claude

- **Code Suggestions**:
  - Generate code that follows the above style and structure.
  - Suggest complete functions or classes, not partial snippets, unless explicitly requested.
  - Ensure suggestions align with C++20 features (e.g., concepts, ranges, coroutines where applicable).
- **Documentation**:
  - Automatically add Doxygen-compatible comments for all generated code.
  - Do not include comments that describe code changes, previous versions, or speculative future changes (e.g., avoid `// Changed to X`, `// Previously Y`, or `// Will add Z later`).
  - Focus on the purpose and behavior of the code as it exists.
- **Refactoring**:
  - Suggest refactored code that improves readability, performance, or maintainability without altering functionality unless requested.
  - Preserve existing documentation unless it violates the “no change-based comments” rule.
- **Error Avoidance**:
  - Ensure generated code compiles without errors under C++20 with GCC/Clang.
  - Avoid suggesting dependencies outside the approved list (Eigen, Cereal, GoogleTest).
- **Test files**: 
  - Include test files in `tests/` for all new features and bug fixes.
  - Make sure the .exe variant is in the `tests/exe/` directory.

## Testing and Validation

- **Unit Tests**: Write tests for every public function and class in `tests/`. Use GoogleTest with clear test case names (e.g., `TensorEngine_ComputeAttention_ValidInput`).
- **Integration Tests**: Include tests in `tests/` for full inference pipelines (e.g., loading a model and generating text).
- **Benchmarks**: Add scripts in `benchmarks/` to measure performance (e.g., tokens/sec, memory usage) on reference hardware.
- **CI/CD**: Configure GitHub Actions for automated builds, tests, and clang-format checks.

## Contribution Guidelines

- **Pull Requests**: Follow a feature-branch workflow. Each PR must include tests and updated documentation.
- **Code Reviews**: Ensure code adheres to style and documentation standards before merging.
- **Issues**: Use GitHub Issues for bug reports and feature requests, with clear templates.

## Example Workflow with Copilot

1. **Task**: Write a function to load a GGUF model file.
2. **Copilot Action**:
   - Suggest a function in `src/model/model_loader.cpp` with a header in `include/turboinfer/model/model_loader.hpp`.
   - Include Doxygen comments describing the function’s purpose, parameters, and return value.
   - Avoid comments like `// Changed to support GGUF format`.
   - Suggest a unit test in `tests/model_loader_test.cpp`.
3. **Output Example**:

   ```cpp
   // include/turboinfer/model/model_loader.hpp
   /**
    * @file model_loader.hpp
    * @brief Defines utilities for loading LLM model files.
    */
   #pragma once
   #include <string>
   #include "turboinfer/core/tensor.hpp"
   
   namespace turboinfer::model {
   /**
    * @brief Loads a GGUF model from a file.
    * @param file_path Path to the GGUF model file.
    * @return Tensor containing the model weights.
    * @throws std::runtime_error if the file is invalid or cannot be read.
    */
   Tensor load_gguf_model(const std::string& file_path);
   }
   ```

## Additional Notes

- **Open-Source Community**: Encourage contributions by providing clear `README.md` and `docs/` guides. Include examples for common use cases (e.g., running inference on a small model).
- **Performance Focus**: Prioritize optimizations like SIMD, multithreading, and memory alignment in Copilot suggestions.
- **Scalability**: Design APIs to support future extensions (e.g., GPU support, distributed inference) without breaking existing code.