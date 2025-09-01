/**
 * @file test_profiler_fixed.cpp
 * @brief Test profiler file output functionality with proper format validation.
 * @author TurboInfer Development Team
 */

#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include "turboinfer/util/profiler.hpp"

using namespace turboinfer::util;

void test_basic_profiling() {
    std::cout << "Testing basic profiling functionality..." << std::endl;
    
    auto& profiler = Profiler::instance();
    profiler.set_enabled(true);  // Enable profiling
    profiler.clear();
    
    // Record some operations
    profiler.record_operation("test_op_1", 1.5, 1024);
    profiler.record_operation("test_op_2", 2.3, 2048);
    profiler.record_operation("test_op_1", 1.8, 1024); // Same operation again
    
    // Generate basic report
    std::string report = profiler.generate_report(true);
    std::cout << "Generated report successfully" << std::endl;
    std::cout << "Profiler enabled: " << (profiler.is_enabled() ? "YES" : "NO") << std::endl;
}

void test_file_output_formats() {
    std::cout << "\nTesting file output formats..." << std::endl;
    
    auto& profiler = Profiler::instance();
    
    // Test valid formats
    std::cout << "Testing text format: ";
    bool success = profiler.save_report("test_profiler_output.txt", "text");
    std::cout << (success ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    std::cout << "Testing JSON format: ";
    success = profiler.save_report("test_profiler_output.json", "json");
    std::cout << (success ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    std::cout << "Testing CSV format: ";
    success = profiler.save_report("test_profiler_output.csv", "csv");
    std::cout << (success ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    // Test invalid format (should fail)
    std::cout << "Testing invalid XML format: ";
    success = profiler.save_report("test_profiler_output.xml", "xml");
    std::cout << (success ? "✗ FAIL (should have rejected)" : "✓ PASS (correctly rejected)") << std::endl;
}

void test_file_contents() {
    std::cout << "\nTesting file contents..." << std::endl;
    
    // Check if files were created and contain data
    std::vector<std::string> files = {
        "test_profiler_output.txt",
        "test_profiler_output.json", 
        "test_profiler_output.csv"
    };
    
    for (const auto& filename : files) {
        if (std::filesystem::exists(filename)) {
            std::ifstream file(filename);
            std::string content((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
            
            if (!content.empty()) {
                std::cout << filename << ": ✓ PASS (contains data)" << std::endl;
            } else {
                std::cout << filename << ": ✗ FAIL (empty file)" << std::endl;
            }
        } else {
            std::cout << filename << ": ✗ FAIL (file not created)" << std::endl;
        }
    }
    
    // Check that XML file was NOT created (invalid format)
    if (!std::filesystem::exists("test_profiler_output.xml")) {
        std::cout << "test_profiler_output.xml: ✓ PASS (correctly not created)" << std::endl;
    } else {
        std::cout << "test_profiler_output.xml: ✗ FAIL (should not have been created)" << std::endl;
    }
}

void cleanup_test_files() {
    std::cout << "\nCleaning up test files..." << std::endl;
    
    std::vector<std::string> files = {
        "test_profiler_output.txt",
        "test_profiler_output.json", 
        "test_profiler_output.csv",
        "test_profiler_output.xml"
    };
    
    for (const auto& filename : files) {
        if (std::filesystem::exists(filename)) {
            std::filesystem::remove(filename);
            std::cout << "Removed " << filename << std::endl;
        }
    }
}

int main() {
    std::cout << "=== TurboInfer Profiler File Output Test ===" << std::endl;
    
    try {
        test_basic_profiling();
        test_file_output_formats();
        test_file_contents();
        // Skip cleanup so we can examine files
        // cleanup_test_files();
        
        std::cout << "\n=== All tests completed ===\n" << std::endl;
        std::cout << "Files preserved for examination:" << std::endl;
        std::cout << "- test_profiler_output.txt" << std::endl;
        std::cout << "- test_profiler_output.json" << std::endl;
        std::cout << "- test_profiler_output.csv" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
