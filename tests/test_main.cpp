/**
 * @file test_main.cpp
 * @brief Main test runner for TurboInfer (manual testing approach).
 * @author J.J.G. Pleunes
 */

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    (void)argc; // Suppress unused parameter warning
    (void)argv; // Suppress unused parameter warning
    
    std::cout << "🚀 TurboInfer Manual Test Runner" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "ℹ️  This is the main test runner file." << std::endl;
    std::cout << "📝 Individual test files are now standalone executables." << std::endl;
    std::cout << std::endl;
    
    std::cout << "🔧 To run individual tests:" << std::endl;
    std::cout << "   cmake --build . --target test_[name]" << std::endl;
    std::cout << "   .\\bin\\test_[name].exe" << std::endl;
    std::cout << std::endl;
    
    std::cout << "📁 Available tests:" << std::endl;
    std::cout << "   • test_library_init    - Library initialization" << std::endl;
    std::cout << "   • test_tensor         - Tensor operations" << std::endl;
    std::cout << "   • test_tensor_engine  - Tensor engine" << std::endl;
    std::cout << "   • test_memory         - Memory management" << std::endl;
    std::cout << "   • test_error_handling - Error handling" << std::endl;
    std::cout << "   • test_performance    - Performance tests" << std::endl;
    std::cout << "   • test_logging        - Logging system" << std::endl;
    std::cout << "   • test_data_types     - Data type tests" << std::endl;
    std::cout << "   • test_tensor_ops     - Tensor operations" << std::endl;
    std::cout << "   • test_model_loader   - Model loading" << std::endl;
    std::cout << "   • test_quantization   - Quantization" << std::endl;
    std::cout << std::endl;
    
    std::cout << "🏃 To run all tests, use: .\\scripts\\run_all_tests.ps1" << std::endl;
    std::cout << "✅ Manual testing approach - no external dependencies required!" << std::endl;
    
    return 0;
}
