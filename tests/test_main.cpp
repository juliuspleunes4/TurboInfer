/**
 * @file test_main.cpp
 * @brief Main entry point for TurboInfer unit tests.
 * @author J.J.G. Pleunes
 */

#include <gtest/gtest.h>
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Running TurboInfer Unit Tests" << std::endl;
    std::cout << "=============================" << std::endl;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
