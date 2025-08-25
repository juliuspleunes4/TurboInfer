#!/usr/bin/env python3
"""
TurboInfer Development Testing Script
Quickly test library functionality and API
"""

import subprocess
import os
import tempfile
import sys

def create_test_program():
    """Create a simple test program to verify TurboInfer functionality"""
    test_code = '''#include "turboinfer/turboinfer.hpp"
#include <iostream>

int main() {
    std::cout << "=== TurboInfer Quick Test ===" << std::endl;
    std::cout << "Build Info: " << turboinfer::build_info() << std::endl;
    
    // Initialize
    if (!turboinfer::initialize(true)) {
        std::cerr << "Failed to initialize TurboInfer" << std::endl;
        return 1;
    }
    
    // Test tensor creation
    try {
        turboinfer::core::TensorShape shape({2, 3});  // Use 2D tensor for now
        turboinfer::core::Tensor tensor(shape, turboinfer::core::DataType::kFloat32);
        
        std::cout << "* Created tensor with " << tensor.shape().ndim() << " dimensions" << std::endl;
        std::cout << "* Total elements: " << tensor.shape().total_size() << std::endl;
        std::cout << "* Memory usage: " << tensor.byte_size() << " bytes" << std::endl;
        
        // Test tensor operations (2D slicing is implemented)
        auto slice_start = std::vector<size_t>{0, 0};
        auto slice_end = std::vector<size_t>{1, 2};
        auto sliced = tensor.slice(slice_start, slice_end);
        std::cout << "* Tensor slicing works" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "X Tensor test failed: " << e.what() << std::endl;
        turboinfer::shutdown();
        return 1;
    }
    
    turboinfer::shutdown();
    std::cout << "* All tests passed!" << std::endl;
    return 0;
}'''
    return test_code

def compile_and_run_test():
    """Compile and run a quick test"""
    print("Creating test program...")
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(create_test_program())
        test_file = f.name
    
    try:
        # Compile
        exe_file = test_file.replace('.cpp', '.exe')
        compile_cmd = [
            'g++', '-std=c++20', '-I', 'include',
            test_file, '-L', 'build/lib', '-lturboinfer',
            '-o', exe_file
        ]
        
        print("Compiling...")
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            return False
        
        # Run
        print("Running test...")
        result = subprocess.run([exe_file], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        
        return result.returncode == 0
        
    finally:
        # Cleanup
        for f in [test_file, exe_file]:
            if os.path.exists(f):
                os.unlink(f)

def main():
    if not os.path.exists('build/lib/libturboinfer.a'):
        print("Error: TurboInfer library not found. Please build first:")
        print("  cmake -B build -S . -G \"MinGW Makefiles\" -DCMAKE_BUILD_TYPE=Debug")
        print("  cmake --build build --parallel")
        sys.exit(1)
    
    if compile_and_run_test():
        print("\n*** TurboInfer is working correctly!")
    else:
        print("\n*** TurboInfer test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
