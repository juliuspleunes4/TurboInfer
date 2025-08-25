cmake_minimum_required(VERSION 3.20)

# Basic inference example
add_executable(basic_inference
    basic_inference.cpp
)

target_link_libraries(basic_inference
    turboinfer
)

# Set target properties
set_target_properties(basic_inference PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Add more examples here as they are created
# add_executable(batch_inference batch_inference.cpp)
# target_link_libraries(batch_inference turboinfer)

# add_executable(quantization_example quantization_example.cpp)
# target_link_libraries(quantization_example turboinfer)

# Install examples
install(TARGETS basic_inference
    RUNTIME DESTINATION bin/examples
)
