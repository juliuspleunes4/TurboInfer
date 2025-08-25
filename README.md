# TurboInfer

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/std/the-standard)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#building-from-source)

**TurboInfer** is a high-performance, open-source C++ library designed to accelerate inference for large language models (LLMs) in production environments. Built with modern C++20, it provides a solid foundation for implementing transformer-based model inference with optimized tensor operations and memory management.

## 🚀 Key Features

- **Modern C++20 Design**: Built with contemporary C++ features including concepts, smart pointers, and RAII
- **High-Performance Tensor System**: Optimized multi-dimensional arrays with efficient memory management  
- **Cross-Platform**: Runs on Windows, Linux, and macOS with CMake build system
- **Extensible Architecture**: Plugin-friendly design for different model formats and optimizations
- **Production Ready**: Professional code structure with comprehensive testing and documentation
- **Zero Dependencies**: Core library requires only C++ standard library (optional: Eigen3, OpenMP)

## 📋 Current Status

**TurboInfer is currently in active development.** The core infrastructure is complete and functional:

### ✅ **Implemented:**
- Complete tensor system with multi-dimensional arrays
- Memory management with RAII patterns
- Build system with CMake and cross-platform scripts
- Logging framework with configurable output
- Comprehensive API design for model loading and inference
- Unit testing framework integration

### 🚧 **In Development:**
- Mathematical operations (GEMM, convolutions, attention)
- Model format parsers (GGUF, SafeTensors, PyTorch, ONNX)
- Quantization algorithms (INT4/INT8)
- GPU acceleration support

## 📋 Requirements

### System Requirements
- **Compiler**: GCC 10+, Clang 12+, or MSVC 2019+ with C++20 support
- **CMake**: Version 3.20 or higher
- **Memory**: Minimum 4GB RAM for development
- **Storage**: Any modern storage (SSD recommended for faster builds)

### Dependencies
- **C++ Standard Library**: C++20 compatible implementation
- **Eigen3**: Linear algebra operations (optional, for optimized math)
- **OpenMP**: Parallel processing (optional, enabled by default)
- **GoogleTest**: Unit testing framework (optional, for development)

## 🛠️ Building from Source

### Quick Development (Recommended)
```powershell
# Windows with PowerShell
.\scripts\dev.ps1 build    # Build the project
.\scripts\dev.ps1 test     # Run tests
.\scripts\dev.ps1 clean    # Clean build artifacts
```

```bash
# Linux/macOS
./scripts/build.sh         # Build the project
python3 tools/test_library.py  # Test the library
```

### Clone the Repository
```bash
git clone https://github.com/juliuspleunes4/TurboInfer.git
cd TurboInfer
```

### Manual Build with CMake (Windows)
```powershell
# Configure the project
cmake -B build -S . -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=20 -DTURBOINFER_BUILD_TESTS=OFF

# Build the library
cmake --build build --config Debug --parallel

# The static library will be created at: build/lib/libturboinfer.a
```

### Manual Build with CMake (Linux/macOS)
```bash
# Create build directory
mkdir build && cd build

# Configure (Release build)
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20

# Build
cmake --build . --config Release --parallel
```

### Build Options
```bash
# Disable tests and examples (recommended for faster builds)
cmake .. -DTURBOINFER_BUILD_TESTS=OFF -DTURBOINFER_BUILD_EXAMPLES=OFF -DTURBOINFER_BUILD_BENCHMARKS=OFF

# Disable OpenMP (if not available)
cmake .. -DTURBOINFER_ENABLE_OPENMP=OFF

# Enable verbose output
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
```

## 🔧 Quick Start

### Basic Library Usage (Current API)
```cpp
#include "turboinfer/turboinfer.hpp"

int main() {
    // Initialize the library
    if (!turboinfer::initialize(true)) {
        std::cerr << "Failed to initialize TurboInfer" << std::endl;
        return 1;
    }
    
    // Display build information
    std::cout << "Build Info: " << turboinfer::build_info() << std::endl;
    
    // Create a tensor
    turboinfer::core::TensorShape shape({2, 3});
    turboinfer::core::Tensor tensor(shape, turboinfer::core::DataType::kFloat32);
    
    std::cout << "Created tensor with " << tensor.shape().ndim() << " dimensions" << std::endl;
    std::cout << "Total elements: " << tensor.shape().total_size() << std::endl;
    std::cout << "Memory usage: " << tensor.byte_size() << " bytes" << std::endl;
    
    // Perform tensor operations
    auto slice_start = std::vector<size_t>{0, 0};
    auto slice_end = std::vector<size_t>{1, 2};
    auto sliced = tensor.slice(slice_start, slice_end);
    
    // Clean up
    turboinfer::shutdown();
    return 0;
}
```

### Compiling Against TurboInfer
```bash
# After building TurboInfer
g++ -std=c++20 -I include your_program.cpp -L build/lib -lturboinfer -o your_program
```

### Future API Design (Under Development)
```cpp
// This API is planned but not yet implemented
#include <turboinfer/turboinfer.hpp>

int main() {
    // Load a pre-trained model (future feature)
    auto model = turboinfer::model::load_model("path/to/model.gguf");
    
    // Create inference engine (future feature)
    turboinfer::model::InferenceEngine engine("path/to/model.gguf");
    
    // Tokenize and generate (future feature)
    auto tokens = engine.encode("Hello, world!");
    auto output = engine.generate(tokens, 50);
    auto result = engine.decode(output);
    
    std::cout << "Generated text: " << result << std::endl;
    return 0;
}
```
## 📚 Documentation

- **[API Reference](docs/)**: Documentation and guides
- **[Status Reports](docs/status/)**: Project progress and current fixes
- **[Development Guide](docs/development.md)**: Setup and contribution guidelines
- **[Examples](examples/)**: Sample applications (coming soon)

## 🏗️ Project Structure

```
TurboInfer/
├── 📁 include/          # Public API headers
│   └── turboinfer/      # Main library headers
├── 📁 src/             # Implementation files
│   ├── core/           # Tensor operations
│   ├── model/          # Model loading/inference
│   ├── optimize/       # Quantization utilities
│   └── util/           # Logging and profiling
├── 📁 scripts/         # Build and development scripts
├── 📁 tools/           # Development utilities
├── 📁 tests/           # Unit tests
├── 📁 docs/            # Documentation
└── 📁 examples/        # Example applications
```

## 🧪 Testing

### Running Tests
```powershell
# Quick test using development script
.\scripts\dev.ps1 test

# Or manual test
python tools/test_library.py
```

### Test Output
```
=== TurboInfer Quick Test ===
Build Info: TurboInfer 1.0.0 (built Aug 25 2025) C++202002 +OpenMP Windows
* Created tensor with 2 dimensions  
* Total elements: 6
* Memory usage: 24 bytes
* Tensor slicing works
* All tests passed!
```

## 🎯 Current Capabilities

### Core Tensor System ✅
- Multi-dimensional tensor creation and management
- Memory-efficient storage with RAII 
- Tensor slicing and reshaping operations
- Support for multiple data types (Float32, Int32, etc.)

### Build System ✅  
- Cross-platform CMake configuration
- MinGW, GCC, Clang, and MSVC support
- Development scripts for quick iteration
- Automated testing framework

### Foundation Architecture ✅
- Modern C++20 codebase with best practices
- Comprehensive logging system
- Plugin-ready design for extensibility
- Professional documentation and examples

### Future Planned Features 🚧
- Mathematical operations (GEMM, attention mechanisms)
- Model format support (GGUF, SafeTensors, PyTorch, ONNX)
- Quantization algorithms (INT4/INT8 compression)
- GPU acceleration (CUDA/OpenCL/Metal)
- Python bindings and language interfaces

## 🤝 Contributing

We welcome contributions from the community! TurboInfer is designed to be a collaborative project.

### How to Contribute
1. **Fork the repository** and create a feature branch
2. **Build the project** using `.\scripts\dev.ps1 build`
3. **Run tests** to ensure everything works: `.\scripts\dev.ps1 test`
4. **Make your changes** following the existing code style
5. **Add tests** for new functionality
6. **Submit a pull request** with a clear description

### Development Setup
```bash
# Clone your fork
git clone https://github.com/juliuspleunes4/TurboInfer.git
cd TurboInfer

# Build and test
.\scripts\dev.ps1 build
.\scripts\dev.ps1 test
```

### Code Style
- Follow C++20 best practices
- Use RAII for resource management
- Include comprehensive documentation
- Add unit tests for new features

## 📞 Issues and Support

- **Bug Reports**: [GitHub Issues](https://github.com/juliuspleunes4/TurboInfer/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/juliuspleunes4/TurboInfer/discussions)
- **Questions**: Check the [documentation](docs/) first

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Eigen Team**: For the excellent linear algebra library
- **GGML/llama.cpp**: Inspiration for model format support
- **Hugging Face**: Model architectures and tokenization approaches
- **OpenAI**: Transformer architecture foundations

## 🗓️ Roadmap

### Phase 1: Foundation Infrastructure ✅ **COMPLETE**
- [x] Core tensor system with multi-dimensional arrays
- [x] Memory management with RAII patterns
- [x] Cross-platform build system (CMake)
- [x] Logging framework with configurable output
- [x] Development scripts and tooling
- [x] Unit testing framework integration

### Phase 2: Mathematical Operations 🚧 **IN PROGRESS**
- [ ] Basic linear algebra (GEMM, matrix operations)
- [ ] Attention mechanisms (self-attention, multi-head)
- [ ] Activation functions (ReLU, GELU, SiLU)
- [ ] Layer normalization and embeddings
- [ ] SIMD optimizations (AVX, NEON)

### Phase 3: Model Loading and Parsing 📋 **PLANNED**
- [ ] GGUF format parser and loader
- [ ] SafeTensors format support
- [ ] PyTorch model conversion utilities
- [ ] ONNX model support
- [ ] Model metadata and configuration handling

### Phase 4: Inference Engine 📋 **PLANNED**
- [ ] Transformer decoder implementation
- [ ] Token generation and sampling
- [ ] KV-cache management
- [ ] Temperature and top-k/top-p sampling
- [ ] Beam search implementation

### Phase 5: Production Features 📋 **FUTURE**
- [ ] Quantization algorithms (INT4/INT8)
- [ ] GPU acceleration (CUDA/ROCm)
- [ ] Batched inference support
- [ ] Streaming text generation
- [ ] Python bindings
- [ ] WebAssembly target
- [ ] Distributed inference

---

**TurboInfer** - Building the foundation for high-performance LLM inference.

*Current Status: Foundation complete, mathematical operations in development*
