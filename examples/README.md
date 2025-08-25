# TurboInfer Examples

This directory contains example programs demonstrating how to use the TurboInfer library.

## Available Examples

### `readme_example.cpp`
Basic example from the main README showing:
- Library initialization
- Tensor creation and basic operations
- Memory usage and dimension queries
- Tensor slicing
- Proper cleanup

**Compile and run:**
```bash
g++ -std=c++20 -I ../include readme_example.cpp -L ../build/lib -lturboinfer -o readme_example
./readme_example
```

## Building Examples

### Using the development script:
```powershell
# Enable examples in build
.\scripts\dev.ps1 build -DTURBOINFER_BUILD_EXAMPLES=ON
```

### Manual compilation:
```bash
# From the examples directory
g++ -std=c++20 -I ../include example_name.cpp -L ../build/lib -lturboinfer -o example_name
```

## Future Examples (Planned)

- **Model Loading**: How to load different model formats
- **Inference Engine**: Running actual text generation  
- **Quantization**: Converting models to lower precision
- **Batched Processing**: Processing multiple inputs efficiently
- **Performance Benchmarking**: Measuring inference speed

## Requirements

- TurboInfer library must be built first
- C++20 compatible compiler
- Access to the built library at `../build/lib/libturboinfer.a`
