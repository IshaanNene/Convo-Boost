cmake_minimum_required(VERSION 3.15)
project(MetalImageProcessing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required frameworks for Metal targets
find_library(FOUNDATION_LIBRARY Foundation)
find_library(METAL_LIBRARY Metal)

# Set compiler flags for Metal targets
set(METAL_COMPILE_OPTIONS
    -fobjc-arc
    -fmodules
    -fcxx-modules
)

# Metal-using targets
add_executable(image_effects_processor image_effects_processor.mm)
target_compile_options(image_effects_processor PRIVATE ${METAL_COMPILE_OPTIONS})
target_link_libraries(image_effects_processor ${FOUNDATION_LIBRARY} ${METAL_LIBRARY})

add_executable(metal_convolution metal_convolution.mm)
target_compile_options(metal_convolution PRIVATE ${METAL_COMPILE_OPTIONS})
target_link_libraries(metal_convolution ${FOUNDATION_LIBRARY} ${METAL_LIBRARY})

# Pure C++ targets (no Metal/Foundation dependencies)
add_executable(random_image_generator random_image_generator.cpp)
add_executable(ppm_to_png ppm_to_png.cpp) 