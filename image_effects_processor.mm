// image_effects_processor.cpp
// Library for applying various image effects using Metal acceleration

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <fstream>
#include "kernels.c"
// Metal shader code as a string - will be compiled at runtime

class ImageEffects {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelineStates;

public:
    ImageEffects() {
        // Initialize Metal
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Failed to create Metal device");
        }

        // Create command queue
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            throw std::runtime_error("Failed to create command queue");
        }

        // Compile shader code
        NSError* error = nil;
        library = [device newLibraryWithSource:[NSString stringWithUTF8String:metalEffectsCode]
                                       options:nil
                                         error:&error];
        if (!library) {
            NSString* errorString = [error localizedDescription];
            throw std::runtime_error(std::string("Failed to create library: ") + 
                                    [errorString UTF8String]);
        }

        // Create compute pipelines for different effects
        createPipelineState("convolution2D");
        createPipelineState("adjustBrightness");
        createPipelineState("adjustContrast");
        createPipelineState("adjustSaturation");
        createPipelineState("edgeDetect");
        createPipelineState("sepiaTone");
        createPipelineState("pixelate");
    }

    // Blur image using Gaussian blur
    void applyGaussianBlur(
        const std::vector<float>& inputImage,
        std::vector<float>& outputImage,
        int width,
        int height,
        int channels,
        float sigma = 1.5f)
    {
        // Generate Gaussian kernel
        int kernelSize = std::max(3, static_cast<int>(std::ceil(sigma * 6)));
        if (kernelSize % 2 == 0) kernelSize++; // Make sure kernel size is odd

        std::vector<float> kernel(kernelSize * kernelSize);
        float sum = 0.0f;

        int radius = kernelSize / 2;
        float twoSigmaSquare = 2.0f * sigma * sigma;

        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                float distance = x * x + y * y;
                float weight = std::exp(-distance / twoSigmaSquare);
                kernel[(y + radius) * kernelSize + (x + radius)] = weight;
                sum += weight;
            }
        }

        // Normalize kernel
        for (auto& value : kernel) {
            value /= sum;
        }

        // Apply convolution with Gaussian kernel
        applyConvolution(inputImage, outputImage, kernel, width, height, channels, kernelSize);
    }

    // Adjust brightness
    void adjustBrightness(
        const std::vector<float>& inputImage,
        std::vector<float>& outputImage,
        int width,
        int height,
        int channels,
        float factor)
    {
        // Create Metal textures for input and output
        MTLTextureDescriptor* textureDescriptor = createTextureDescriptor(width, height);
        id<MTLTexture> inputTexture = [device newTextureWithDescriptor:textureDescriptor];
        id<MTLTexture> outputTexture = [device newTextureWithDescriptor:textureDescriptor];

        // Copy input data to texture
        copyDataToTexture(inputImage, inputTexture, width, height, channels);

        // Create buffer for brightness factor
        id<MTLBuffer> factorBuffer = [device newBufferWithBytes:&factor
                                                         length:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        [computeEncoder setComputePipelineState:pipelineStates["adjustBrightness"]];
        [computeEncoder setTexture:inputTexture atIndex:0];
        [computeEncoder setTexture:outputTexture atIndex:1];
        [computeEncoder setBuffer:factorBuffer offset:0 atIndex:0];

        dispatchCompute(computeEncoder, width, height);
        [computeEncoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy output data back to CPU
        outputImage.resize(width * height * channels);
        copyDataFromTexture(outputTexture, outputImage, width, height, channels);
    }

    // Adjust contrast
    void adjustContrast(
        const std::vector<float>& inputImage,
        std::vector<float>& outputImage,
        int width,
        int height,
        int channels,
        float factor)
    {
        // Create Metal textures for input and output
        MTLTextureDescriptor* textureDescriptor = createTextureDescriptor(width, height);
        id<MTLTexture> inputTexture = [device newTextureWithDescriptor:textureDescriptor];
        id<MTLTexture> outputTexture = [device newTextureWithDescriptor:textureDescriptor];

        // Copy input data to texture
        copyDataToTexture(inputImage, inputTexture, width, height, channels);

        // Create buffer for contrast factor
        id<MTLBuffer> factorBuffer = [device newBufferWithBytes:&factor
                                                         length:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        [computeEncoder setComputePipelineState:pipelineStates["adjustContrast"]];
        [computeEncoder setTexture:inputTexture atIndex:0];
        [computeEncoder setTexture:outputTexture atIndex:1];
        [computeEncoder setBuffer:factorBuffer offset:0 atIndex:0];

        dispatchCompute(computeEncoder, width, height);
        [computeEncoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy output data back to CPU
        outputImage.resize(width * height * channels);
        copyDataFromTexture(outputTexture, outputImage, width, height, channels);
    }

    // Adjust saturation
    void adjustSaturation(
        const std::vector<float>& inputImage,
        std::vector<float>& outputImage,
        int width,
        int height,
        int channels,
        float factor)
    {
        // Create Metal textures for input and output
        MTLTextureDescriptor* textureDescriptor = createTextureDescriptor(width, height);
        id<MTLTexture> inputTexture = [device newTextureWithDescriptor:textureDescriptor];
        id<MTLTexture> outputTexture = [device newTextureWithDescriptor:textureDescriptor];

        // Copy input data to texture
        copyDataToTexture(inputImage, inputTexture, width, height, channels);

        // Create buffer for saturation factor
        id<MTLBuffer> factorBuffer = [device newBufferWithBytes:&factor
                                                         length:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        [computeEncoder setComputePipelineState:pipelineStates["adjustSaturation"]];
        [computeEncoder setTexture:inputTexture atIndex:0];
        [computeEncoder setTexture:outputTexture atIndex:1];
        [computeEncoder setBuffer:factorBuffer offset:0 atIndex:0];

        dispatchCompute(computeEncoder, width, height);
        [computeEncoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy output data back to CPU
        outputImage.resize(width * height * channels);
        copyDataFromTexture(outputTexture, outputImage, width, height, channels);
    }

    // Apply edge detection
    void applyEdgeDetection(
        const std::vector<float>& inputImage,
        std::vector<float>& outputImage,
        int width,
        int height,
        int channels)
    {
        // Create Metal textures for input and output
        MTLTextureDescriptor* textureDescriptor = createTextureDescriptor(width, height);
        id<MTLTexture> inputTexture = [device newTextureWithDescriptor:textureDescriptor];
        id<MTLTexture> outputTexture = [device newTextureWithDescriptor:textureDescriptor];

        // Copy input data to texture
        copyDataToTexture(inputImage, inputTexture, width, height, channels);

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        [computeEncoder setComputePipelineState:pipelineStates["edgeDetect"]];
        [computeEncoder setTexture:inputTexture atIndex:0];
        [computeEncoder setTexture:outputTexture atIndex:1];

       dispatchCompute(computeEncoder, width, height);
        [computeEncoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy output data back to CPU
        outputImage.resize(width * height * channels);
        copyDataFromTexture(outputTexture, outputImage, width, height, channels);
    }

    // Apply sepia tone effect
    void applySepiaEffect(
        const std::vector<float>& inputImage,
        std::vector<float>& outputImage,
        int width,
        int height,
        int channels,
        float intensity = 1.0f)
    {
        // Create Metal textures for input and output
        MTLTextureDescriptor* textureDescriptor = createTextureDescriptor(width, height);
        id<MTLTexture> inputTexture = [device newTextureWithDescriptor:textureDescriptor];
        id<MTLTexture> outputTexture = [device newTextureWithDescriptor:textureDescriptor];

        // Copy input data to texture
        copyDataToTexture(inputImage, inputTexture, width, height, channels);

        // Create buffer for intensity factor
        id<MTLBuffer> intensityBuffer = [device newBufferWithBytes:&intensity
                                                         length:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        [computeEncoder setComputePipelineState:pipelineStates["sepiaTone"]];
        [computeEncoder setTexture:inputTexture atIndex:0];
        [computeEncoder setTexture:outputTexture atIndex:1];
        [computeEncoder setBuffer:intensityBuffer offset:0 atIndex:0];

        dispatchCompute(computeEncoder, width, height);
        [computeEncoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy output data back to CPU
        outputImage.resize(width * height * channels);
        copyDataFromTexture(outputTexture, outputImage, width, height, channels);
    }

    // Apply pixelate effect
    void applyPixelateEffect(
        const std::vector<float>& inputImage,
        std::vector<float>& outputImage,
        int width,
        int height,
        int channels,
        int blockSize = 8)
    {
        // Create Metal textures for input and output
        MTLTextureDescriptor* textureDescriptor = createTextureDescriptor(width, height);
        id<MTLTexture> inputTexture = [device newTextureWithDescriptor:textureDescriptor];
        id<MTLTexture> outputTexture = [device newTextureWithDescriptor:textureDescriptor];

        // Copy input data to texture
        copyDataToTexture(inputImage, inputTexture, width, height, channels);

        // Create buffer for block size
        id<MTLBuffer> blockSizeBuffer = [device newBufferWithBytes:&blockSize
                                                         length:sizeof(int)
                                                        options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        [computeEncoder setComputePipelineState:pipelineStates["pixelate"]];
        [computeEncoder setTexture:inputTexture atIndex:0];
        [computeEncoder setTexture:outputTexture atIndex:1];
        [computeEncoder setBuffer:blockSizeBuffer offset:0 atIndex:0];

        dispatchCompute(computeEncoder, width, height);
        [computeEncoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy output data back to CPU
        outputImage.resize(width * height * channels);
        copyDataFromTexture(outputTexture, outputImage, width, height, channels);
    }

    // Apply sharpen effect
    void applySharpen(
        const std::vector<float>& inputImage,
        std::vector<float>& outputImage,
        int width,
        int height,
        int channels,
        float strength = 1.0f)
    {
        // Sharpen kernel
        std::vector<float> kernel = {
            0.0f, -1.0f * strength, 0.0f,
            -1.0f * strength, 1.0f + 4.0f * strength, -1.0f * strength,
            0.0f, -1.0f * strength, 0.0f
        };

        applyConvolution(inputImage, outputImage, kernel, width, height, channels, 3);
    }

    // Apply emboss effect
    void applyEmboss(
        const std::vector<float>& inputImage,
        std::vector<float>& outputImage,
        int width,
        int height,
        int channels)
    {
        // Emboss kernel
        std::vector<float> kernel = {
            -2.0f, -1.0f, 0.0f,
            -1.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 2.0f
        };

        applyConvolution(inputImage, outputImage, kernel, width, height, channels, 3);
    }

    // Apply general convolution (foundation for many effects)
    void applyConvolution(
        const std::vector<float>& inputImage,
        std::vector<float>& outputImage,
        const std::vector<float>& kernel,
        int width,
        int height,
        int channels,
        int kernelSize)
    {
        // Create Metal textures for input and output
        MTLTextureDescriptor* textureDescriptor = createTextureDescriptor(width, height);
        id<MTLTexture> inputTexture = [device newTextureWithDescriptor:textureDescriptor];
        id<MTLTexture> outputTexture = [device newTextureWithDescriptor:textureDescriptor];

        // Copy input data to texture
        copyDataToTexture(inputImage, inputTexture, width, height, channels);

        // Create buffer for convolution kernel
        id<MTLBuffer> kernelBuffer = [device newBufferWithBytes:kernel.data()
                                                         length:kernel.size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];

        // Create buffer for kernel size
        id<MTLBuffer> kernelSizeBuffer = [device newBufferWithBytes:&kernelSize
                                                             length:sizeof(int)
                                                            options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        [computeEncoder setComputePipelineState:pipelineStates["convolution2D"]];
        [computeEncoder setTexture:inputTexture atIndex:0];
        [computeEncoder setTexture:outputTexture atIndex:1];
        [computeEncoder setBuffer:kernelBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:kernelSizeBuffer offset:0 atIndex:1];

        dispatchCompute(computeEncoder, width, height);
        [computeEncoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy output data back to CPU
        outputImage.resize(width * height * channels);
        copyDataFromTexture(outputTexture, outputImage, width, height, channels);
    }

private:
    // Helper function to create pipeline states
    void createPipelineState(const std::string& functionName) {
        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:functionName.c_str()]];
        if (!function) {
            throw std::runtime_error("Failed to find function: " + functionName);
        }

        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipelineState) {
            NSString* errorString = [error localizedDescription];
            throw std::runtime_error(std::string("Failed to create pipeline state for ") + 
                                    functionName + ": " + [errorString UTF8String]);
        }

        pipelineStates[functionName] = pipelineState;
    }

    // Helper function to create texture descriptor
    MTLTextureDescriptor* createTextureDescriptor(int width, int height) {
        MTLTextureDescriptor* textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                                     width:width
                                                                                                    height:height
                                                                                                 mipmapped:NO];
        textureDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        return textureDescriptor;
    }

    // Helper function to copy data to texture
    void copyDataToTexture(const std::vector<float>& data, id<MTLTexture> texture, int width, int height, int channels) {
        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        [texture replaceRegion:region
                    mipmapLevel:0
                      withBytes:data.data()
                    bytesPerRow:width * channels * sizeof(float)];
    }

    // Helper function to copy data from texture
    void copyDataFromTexture(id<MTLTexture> texture, std::vector<float>& data, int width, int height, int channels) {
        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        [texture getBytes:data.data()
             bytesPerRow:width * channels * sizeof(float)
              fromRegion:region
             mipmapLevel:0];
    }

    // Helper function to dispatch compute kernel
    void dispatchCompute(id<MTLComputeCommandEncoder> encoder, int width, int height) {
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadGroups = MTLSizeMake(
            (width + threadGroupSize.width - 1) / threadGroupSize.width,
            (height + threadGroupSize.height - 1) / threadGroupSize.height,
            1
        );

        [encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupSize];
    }
};

// Utility function to save an image to a PPM file
bool saveImageToPPM(const std::string& filename, const std::vector<float>& image, int width, int height, int channels = 4) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // Write PPM header
    file << "P6\n" << width << " " << height << "\n255\n";

    // Write image data (convert float values to bytes)
    std::vector<unsigned char> byteData(width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) { // Only RGB for PPM
                float value = image[(y * width + x) * channels + c];
                // Clamp and convert to byte
                byteData[(y * width + x) * 3 + c] = static_cast<unsigned char>(std::min(std::max(value * 255.0f, 0.0f), 255.0f));
            }
        }
    }

    file.write(reinterpret_cast<const char*>(byteData.data()), byteData.size());
    return file.good();
}

// Function to load an image from a PPM file
std::vector<float> loadImageFromPPM(const std::string& filename, int& width, int& height, int channels = 4) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }

    // Read PPM header
    std::string format;
    file >> format;
    if (format != "P6") {
        std::cerr << "Unsupported format: " << format << std::endl;
        return {};
    }

    file >> width >> height;

    int maxVal;
    file >> maxVal;

    // Skip the newline character
    file.get();

    // Read binary data
    std::vector<unsigned char> byteData(width * height * 3);
    file.read(reinterpret_cast<char*>(byteData.data()), byteData.size());

    // Convert to float (with additional alpha channel if requested)
    std::vector<float> result(width * height * channels, 1.0f); // Initialize with alpha = 1.0
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) { // Only RGB from PPM
                result[(y * width + x) * channels + c] = byteData[(y * width + x) * 3 + c] / 255.0f;
            }
        }
    }

    return result;
}

int main() {
    try {
        ImageEffects effects;

        // Parameters
        const int width = 512;
        const int height = 512;
        const int channels = 4; // RGBA

        // Load test image or generate one
        int loadedWidth, loadedHeight;
        std::vector<float> inputImage;

        // Try to load an image, or generate one if loading fails
        inputImage = loadImageFromPPM("testcard.ppm", loadedWidth, loadedHeight, channels);

        if (inputImage.empty()) {
            std::cout << "Generating a test image..." << std::endl;
            // Generate a simple gradient image
            inputImage.resize(width * height * channels);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float normalizedX = static_cast<float>(x) / width;
                    float normalizedY = static_cast<float>(y) / height;

                    inputImage[(y * width + x) * channels + 0] = normalizedX; // Red
                    inputImage[(y * width + x) * channels + 1] = normalizedY; // Green
                    inputImage[(y * width + x) * channels + 2] = (normalizedX + normalizedY) / 2.0f; // Blue
                    inputImage[(y * width + x) * channels + 3] = 1.0f; // Alpha
                }
            }

            loadedWidth = width;
            loadedHeight = height;

            // Save the generated image
            saveImageToPPM("original.ppm", inputImage, loadedWidth, loadedHeight, channels);
        }

        std::cout << "Applying effects to image (" << loadedWidth << "x" << loadedHeight << ")..." << std::endl;

        // Container for output
        std::vector<float> outputImage;

        // Apply various effects and save the results

        // 1. Gaussian blur
        effects.applyGaussianBlur(inputImage, outputImage, loadedWidth, loadedHeight, channels, 3.0f);
        saveImageToPPM("effect_blur.ppm", outputImage, loadedWidth, loadedHeight, channels);
        std::cout << "Generated effect_blur.ppm" << std::endl;

        // 2. Brightness adjustment
        effects.adjustBrightness(inputImage, outputImage, loadedWidth, loadedHeight, channels, 1.5f);
        saveImageToPPM("effect_brightness.ppm", outputImage, loadedWidth, loadedHeight, channels);
        std::cout << "Generated effect_brightness.ppm" << std::endl;

        // 3. Contrast adjustment
        effects.adjustContrast(inputImage, outputImage, loadedWidth, loadedHeight, channels, 2.0f);
        saveImageToPPM("effect_contrast.ppm", outputImage, loadedWidth, loadedHeight, channels);
        std::cout << "Generated effect_contrast.ppm" << std::endl;

        // 4. Saturation adjustment
        effects.adjustSaturation(inputImage, outputImage, loadedWidth, loadedHeight, channels, 1.5f);
        saveImageToPPM("effect_saturation.ppm", outputImage, loadedWidth, loadedHeight, channels);
        std::cout << "Generated effect_saturation.ppm" << std::endl;

        // 5. Edge detection
        effects.applyEdgeDetection(inputImage, outputImage, loadedWidth, loadedHeight, channels);
        saveImageToPPM("effect_edge.ppm", outputImage, loadedWidth, loadedHeight, channels);
        std::cout << "Generated effect_edge.ppm" << std::endl;

        // 6. Sepia tone
        effects.applySepiaEffect(inputImage, outputImage, loadedWidth, loadedHeight, channels, 1.0f);
        saveImageToPPM("effect_sepia.ppm", outputImage, loadedWidth, loadedHeight, channels);
        std::cout << "Generated effect_sepia.ppm" << std::endl;

        // 7. Pixelate
        effects.applyPixelateEffect(inputImage, outputImage, loadedWidth, loadedHeight, channels, 8);
        saveImageToPPM("effect_pixelate.ppm", outputImage, loadedWidth, loadedHeight, channels);
        std::cout << "Generated effect_pixelate.ppm" << std::endl;

        // 8. Sharpen
        effects.applySharpen(inputImage, outputImage, loadedWidth, loadedHeight, channels, 2.0f);
        saveImageToPPM("effect_sharpen.ppm", outputImage, loadedWidth, loadedHeight, channels);
        std::cout << "Generated effect_sharpen.ppm" << std::endl;

        // 9. Emboss
        effects.applyEmboss(inputImage, outputImage, loadedWidth, loadedHeight, channels);
        saveImageToPPM("effect_emboss.ppm", outputImage, loadedWidth, loadedHeight, channels);
        std::cout << "Generated effect_emboss.ppm" << std::endl;

        std::cout << "All effects applied successfully!" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
