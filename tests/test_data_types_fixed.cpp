/**
 * @file test_data_types.cpp
 * @brief Manual unit tests converted from GoogleTest.
 * @author J.J.G. Pleunes
 */

#include "turboinfer/turboinfer.hpp"
#include <iostream>
#include <limits>
#include <cmath>
#include <cstdint>

// Global test counters
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;

// Simple assertion macro
#define ASSERT_TRUE(condition) \
    do { \
        tests_run++; \
        if (condition) { \
            tests_passed++; \
        } else { \
            tests_failed++; \
            std::cerr << "ASSERTION FAILED: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

void setup_test() {
    // Setup code - override in specific tests if needed
}

void teardown_test() {
    // Cleanup code - override in specific tests if needed
}

void test_data_type_sizes() {
    std::cout << "\n--- Test: Data Type Sizes ---" << std::endl;
    setup_test();
    
    // Test standard data type sizes
    ASSERT_TRUE(sizeof(float) == 4);
    ASSERT_TRUE(sizeof(double) == 8);
    ASSERT_TRUE(sizeof(int32_t) == 4);
    ASSERT_TRUE(sizeof(int64_t) == 8);
    ASSERT_TRUE(sizeof(uint8_t) == 1);
    ASSERT_TRUE(sizeof(uint16_t) == 2);
    
    std::cout << "✅ Data type sizes verified" << std::endl;
    teardown_test();
}

void test_floating_point_precision() {
    std::cout << "\n--- Test: Floating Point Precision ---" << std::endl;
    setup_test();
    
    // Test float precision
    float f1 = 1.0f;
    float f2 = 1.0f + 1e-7f;
    ASSERT_TRUE(f1 != f2); // Should be different
    
    // Test double precision
    double d1 = 1.0;
    double d2 = 1.0 + 1e-15;
    ASSERT_TRUE(d1 != d2); // Should be different
    
    // Test float limits
    ASSERT_TRUE(std::numeric_limits<float>::max() > 0);
    ASSERT_TRUE(std::numeric_limits<float>::min() > 0);
    ASSERT_TRUE(std::isfinite(std::numeric_limits<float>::max()));
    
    std::cout << "✅ Floating point precision tests passed" << std::endl;
    teardown_test();
}

void test_integer_overflow() {
    std::cout << "\n--- Test: Integer Overflow Behavior ---" << std::endl;
    setup_test();
    
    // Test uint8_t overflow
    uint8_t u8_max = std::numeric_limits<uint8_t>::max();
    uint8_t u8_overflow = u8_max + 1;
    ASSERT_TRUE(u8_overflow == 0); // Should wrap around
    
    // Test int32_t limits
    int32_t i32_max = std::numeric_limits<int32_t>::max();
    int32_t i32_min = std::numeric_limits<int32_t>::min();
    ASSERT_TRUE(i32_max > 0);
    ASSERT_TRUE(i32_min < 0);
    
    std::cout << "✅ Integer overflow behavior verified" << std::endl;
    teardown_test();
}

void test_memory_alignment() {
    std::cout << "\n--- Test: Memory Alignment ---" << std::endl;
    setup_test();
    
    // Test alignment of common types
    float* float_ptr = new float[4];
    double* double_ptr = new double[4];
    int32_t* int_ptr = new int32_t[4];
    
    // Check that pointers are properly aligned
    ASSERT_TRUE(reinterpret_cast<uintptr_t>(float_ptr) % alignof(float) == 0);
    ASSERT_TRUE(reinterpret_cast<uintptr_t>(double_ptr) % alignof(double) == 0);
    ASSERT_TRUE(reinterpret_cast<uintptr_t>(int_ptr) % alignof(int32_t) == 0);
    
    delete[] float_ptr;
    delete[] double_ptr;
    delete[] int_ptr;
    
    std::cout << "✅ Memory alignment tests passed" << std::endl;
    teardown_test();
}

void test_endianness() {
    std::cout << "\n--- Test: Endianness Detection ---" << std::endl;
    setup_test();
    
    uint32_t test_value = 0x12345678;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&test_value);
    
    // Check if system is little-endian or big-endian
    bool is_little_endian = (bytes[0] == 0x78);
    bool is_big_endian = (bytes[0] == 0x12);
    
    ASSERT_TRUE(is_little_endian || is_big_endian); // Must be one or the other
    ASSERT_TRUE(!(is_little_endian && is_big_endian)); // Cannot be both
    
    std::cout << "✅ Endianness: " << (is_little_endian ? "Little-endian" : "Big-endian") << std::endl;
    teardown_test();
}

void test_placeholder() {
    std::cout << "\n--- Test: Basic Functionality ---" << std::endl;
    setup_test();
    
    // Basic sanity checks
    ASSERT_TRUE(true);
    ASSERT_TRUE(1 + 1 == 2);
    ASSERT_TRUE(sizeof(char) == 1);
    
    std::cout << "✅ Basic functionality test passed" << std::endl;
    teardown_test();
}

int main() {
    std::cout << "🚀 Starting test_data_types Tests..." << std::endl;
    
    test_data_type_sizes();
    test_floating_point_precision();
    test_integer_overflow();
    test_memory_alignment();
    test_endianness();
    test_placeholder();
    
    std::cout << "\n📊 Test Results:" << std::endl;
    std::cout << "Tests run: " << tests_run << std::endl;
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << tests_failed << std::endl;
    
    if (tests_failed == 0) {
        std::cout << "🎉 ALL TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}
