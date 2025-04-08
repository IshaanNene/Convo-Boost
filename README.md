# Metal Image Processing

This project demonstrates GPU-accelerated image processing using Apple's Metal framework. It includes four main components:

1. **Image Effects Processor**: Applies various effects to images (blur, brightness, contrast, etc.)
2. **Metal Convolution**: Demonstrates 2D convolution operations on images
3. **Random Image Generator**: Creates test images for processing
4. **PPM to PNG Converter**: Converts PPM images to PNG format for easier viewing and sharing

## Requirements

- macOS (tested on macOS 12.0+)
- Xcode Command Line Tools
- CMake (3.15+)
- C++17 compatible compiler

## Building the Project

```bash
# Clone the repository
git clone https://github.com/IshaanNene/Convo-Boost.git
cd Convo-Boost

# Run the build script
./build.sh
```

Or manually:

```bash
# Create a build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
make
```

## Running the Programs

### Random Image Generator

This utility creates test images that can be used with the other programs:

```bash
./random_image_generator
```

This will generate several test images in PPM format:
- noise.ppm
- gradient.ppm
- checkerboard.ppm
- circular.ppm
- testcard.ppm

### Image Effects Processor

This program applies various effects to an input image:

```bash
./image_effects_processor
```

By default, it will try to load "testcard.ppm" or generate a test image if the file doesn't exist. It will then apply several effects and save the results as:
- effect_blur.ppm
- effect_brightness.ppm
- effect_contrast.ppm
- effect_saturation.ppm
- effect_edge.ppm
- effect_sepia.ppm
- effect_pixelate.ppm
- effect_sharpen.ppm
- effect_emboss.ppm

### Metal Convolution

This program demonstrates the performance difference between CPU and GPU-based convolution:

```bash
./metal_convolution
```

It will run convolution operations on images of different sizes and report the performance comparison.

### PPM to PNG Converter

This utility converts PPM images to PNG format for easier viewing and sharing:

```bash
./ppm_to_png <directory_path>
```

This will convert all PPM files in the specified directory to PNG format. For example:

```bash
# Convert all PPM files in the current directory
./ppm_to_png .

# Convert all PPM files in a specific directory
./ppm_to_png /path/to/images
```

## Viewing the Results

The output images are initially in PPM format. You can:

1. Use the included PPM to PNG converter to convert them to PNG format
2. View them using various image viewers that support PPM format
3. Convert them to other formats using tools like ImageMagick:

```bash
# Install ImageMagick if needed
brew install imagemagick

# Convert PPM to PNG
convert input.ppm output.png
```

## Performance

The Metal-based image processing demonstrates significant performance improvements over CPU-based processing:

- For 512x512 images: ~106x speedup
- For 1024x1024 images: ~163x speedup
- For 2048x2048 images: ~109x speedup

## License

This project is licensed under the MIT License - see the LICENSE file for details. 