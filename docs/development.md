# TurboInfer Development Guide

## Project Structure

```
TurboInfer/
├── include/turboinfer/          # Public headers
│   ├── core/                    # Core tensor and engine classes
│   ├── model/                   # Model loading and inference
│   ├── optimize/                # Optimization utilities
│   ├── util/                    # Utility classes
│   └── turboinfer.hpp          # Main header
├── src/                        # Implementation files
│   ├── core/                   # Core implementations
│   ├── model/                  # Model implementations
│   ├── optimize/               # Optimization implementations
│   ├── util/                   # Utility implementations
│   └── turboinfer.cpp          # Main library implementation
├── tests/                      # Unit tests
├── examples/                   # Example applications
├── benchmarks/                 # Performance benchmarks
├── docs/                       # Documentation
├── cmake/                      # CMake modules
└── .github/                    # GitHub workflows and configs
```

## Building the Project

### Prerequisites

- **C++ Compiler**: GCC 10+, Clang 12+, or MSVC 2019+ with C++20 support
- **CMake**: Version 3.20 or higher
- **Dependencies**:
  - Eigen3 (for linear algebra)
  - OpenMP (optional, for parallelization)
  - GoogleTest (optional, for testing)

### Windows Build

```powershell
# Using the provided build script
.\build.ps1 Release -Tests -Examples

# Or manually with CMake
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Linux/macOS Build

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Run tests
ctest
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `TURBOINFER_BUILD_TESTS` | ON | Build unit tests |
| `TURBOINFER_BUILD_EXAMPLES` | ON | Build example applications |
| `TURBOINFER_BUILD_BENCHMARKS` | ON | Build performance benchmarks |
| `TURBOINFER_ENABLE_SIMD` | ON | Enable SIMD optimizations |
| `TURBOINFER_ENABLE_OPENMP` | ON | Enable OpenMP support |

## Core Architecture

### Tensor System

The tensor system is built around the `Tensor` class which provides:

- Multi-dimensional array storage
- Multiple data types (float32, int8, etc.)
- Memory management with RAII
- Basic operations (reshape, slice, clone)

```cpp
#include <turboinfer/core/tensor.hpp>

// Create a 2x3 float tensor
turboinfer::TensorShape shape{2, 3};
turboinfer::Tensor tensor(shape, turboinfer::DataType::kFloat32);

// Fill with data
tensor.fill<float>(1.0f);

// Access data
float* data = tensor.data_ptr<float>();
```

### Tensor Engine

The `TensorEngine` provides optimized implementations of:

- Matrix operations (matmul, batched operations)
- Activation functions (ReLU, GELU, SiLU)
- Attention mechanisms
- Normalization (LayerNorm, RMSNorm)
- Element-wise operations

```cpp
#include <turboinfer/core/tensor_engine.hpp>

turboinfer::TensorEngine engine(turboinfer::ComputeDevice::kCPU);

// Perform matrix multiplication
auto result = engine.matmul(tensor_a, tensor_b);
```

### Model Loading

The model loading system supports multiple formats:

- GGUF (GGML Universal Format)
- SafeTensors
- PyTorch (.pt/.pth)
- ONNX

```cpp
#include <turboinfer/model/model_loader.hpp>

// Load model from file
auto model_data = turboinfer::ModelLoader::load("model.gguf");

// Access model metadata
const auto& metadata = model_data.metadata();
std::cout << "Model: " << metadata.name << std::endl;
```

### Inference Engine

The `InferenceEngine` provides high-level text generation:

```cpp
#include <turboinfer/model/inference_engine.hpp>

// Create inference engine
turboinfer::InferenceConfig config;
config.temperature = 0.8f;
config.max_sequence_length = 2048;

turboinfer::InferenceEngine engine("model.gguf", config);

// Generate text
auto result = engine.generate("Hello, world!", 50);
std::string generated = engine.decode(result.tokens);
```

## Development Workflow

### Adding New Features

1. **Design**: Create or update header files in `include/turboinfer/`
2. **Implement**: Add implementation in corresponding `src/` directory
3. **Test**: Create unit tests in `tests/`
4. **Document**: Update documentation and examples
5. **Benchmark**: Add performance tests if applicable

### Code Style

- Use C++20 features appropriately
- Follow RAII principles
- Prefer smart pointers over raw pointers
- Use meaningful variable and function names
- Add comprehensive documentation comments

### Testing

All public APIs should have corresponding unit tests:

```cpp
#include <gtest/gtest.h>
#include "turboinfer/core/tensor.hpp"

TEST(TensorTest, BasicConstruction) {
    turboinfer::TensorShape shape{2, 3};
    turboinfer::Tensor tensor(shape);
    
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_FALSE(tensor.empty());
}
```

### Performance Considerations

- Use SIMD instructions where beneficial
- Leverage OpenMP for parallelization
- Minimize memory allocations in hot paths
- Profile regularly with the built-in profiler

```cpp
#include <turboinfer/util/profiler.hpp>

void my_function() {
    TURBOINFER_PROFILE_FUNCTION();
    // Function implementation
}
```

## Implementation Status

### ✅ Completed

- Basic project structure
- Tensor class with shape management
- Build system with CMake
- Unit testing framework
- Logging system
- Basic examples

### 🚧 In Progress

- Tensor operations (mathematical functions)
- Model loading (GGUF format)
- Inference engine core logic
- Quantization support

### 📋 Planned

- GPU acceleration (CUDA/ROCm)
- Advanced optimizations
- Python bindings
- Distributed inference
- WebAssembly support

## Contributing

### Setting Up Development Environment

1. Clone the repository
2. Install dependencies (Eigen3, OpenMP)
3. Build the project: `.\build.ps1 Debug -Tests -Examples`
4. Run tests: `cd build && ctest`

### Submitting Changes

1. Create a feature branch
2. Implement changes with tests
3. Ensure all tests pass
4. Update documentation
5. Submit pull request

### Code Review Process

- All changes require review
- Tests must pass CI
- Documentation must be updated
- Performance impact should be measured

## Debugging

### Common Issues

1. **CMake Configuration Fails**
   - Check CMake version (>= 3.20)
   - Ensure Eigen3 is installed
   - Verify compiler supports C++20

2. **Build Errors**
   - Check compiler version
   - Ensure all dependencies are found
   - Try clean build

3. **Runtime Errors**
   - Enable debug builds for better error messages
   - Use the logging system
   - Check tensor dimensions compatibility

### Debug Builds

```powershell
.\build.ps1 Debug -Verbose
```

Debug builds include:
- Debug symbols
- Additional runtime checks
- Detailed logging
- Memory debugging (when available)

## Resources

- [Eigen Documentation](https://eigen.tuxfamily.org/dox/)
- [CMake Documentation](https://cmake.org/documentation/)
- [GoogleTest Primer](https://google.github.io/googletest/primer.html)
- [C++20 Reference](https://en.cppreference.com/w/cpp/20)
