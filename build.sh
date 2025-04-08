#!/bin/bash

# Exit on error
set -e

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..

# Build the project
echo "Building the project..."
make -j$(sysctl -n hw.ncpu)

echo "Build completed successfully!"
echo ""
echo "To run the random image generator:"
echo "  ./random_image_generator"
echo ""
echo "To run the image effects processor:"
echo "  ./image_effects_processor"
echo ""
echo "To run the metal convolution demo:"
echo "  ./metal_convolution"
echo ""
echo "Note: The programs will generate PPM image files that you can view with an image viewer or convert to other formats." 