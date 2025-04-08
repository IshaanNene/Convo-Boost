// Metal Convolution Implementation in C++
// This demonstrates how to accelerate 2D image convolutions on Apple M1 GPUs

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <chrono>

// Metal shader code as a string - will be compiled at runtime
const char* metalShaderCode = R"(
#include <metal_stdlib>
using namespace metal;

kernel void convolution2D(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float* convKernel [[buffer(0)]],
    constant int& kernelSize [[buffer(1)]],
    uint2 pos [[thread_position_in_grid]])
{
    const int kernelRadius = kernelSize / 2;
    const int width = inputTexture.get_width();
    const int height = inputTexture.get_height();
    
    // Check if we're within the image bounds
    if (pos.x >= width || pos.y >= height) {
        return;
    }
    
    float4 result = float4(0.0, 0.0, 0.0, 0.0);
    
    // Apply the convolution
    for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
            int2 samplePos;
            samplePos.x = pos.x + kx - kernelRadius;
            samplePos.y = pos.y + ky - kernelRadius;
            
            // Clamp to edge for boundary pixels
            samplePos.x = clamp(samplePos.x, 0, width - 1);
            samplePos.y = clamp(samplePos.y, 0, height - 1);
            
            float4 sample = inputTexture.read(uint2(samplePos));
            float kernelValue = convKernel[ky * kernelSize + kx];
            
            result += sample * kernelValue;
        }
    }
    
    outputTexture.write(result, pos);
}
)";

class MetalConvolution {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLFunction> kernelFunction;
    id<MTLComputePipelineState> pipelineState;
    
public:
    MetalConvolution() {
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
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:metalShaderCode]
                                                      options:nil
                                                        error:&error];
        if (!library) {
            NSString* errorString = [error localizedDescription];
            throw std::runtime_error(std::string("Failed to create library: ") + 
                                    [errorString UTF8String]);
        }
        
        // Get kernel function
        kernelFunction = [library newFunctionWithName:@"convolution2D"];
        if (!kernelFunction) {
            throw std::runtime_error("Failed to create kernel function");
        }
        
        // Create compute pipeline
        pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pipelineState) {
            NSString* errorString = [error localizedDescription];
            throw std::runtime_error(std::string("Failed to create pipeline state: ") + 
                                    [errorString UTF8String]);
        }
    }
    
    void applyConvolution(
        const std::vector<float>& inputImage,
        std::vector<float>& outputImage,
        const std::vector<float>& kernel,
        int imageWidth,
        int imageHeight,
        int channels,
        int kernelSize
    ) {
        // Create Metal textures for input and output
        MTLTextureDescriptor* textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                                     width:imageWidth
                                                                                                    height:imageHeight
                                                                                                 mipmapped:NO];
        textureDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        
        id<MTLTexture> inputTexture = [device newTextureWithDescriptor:textureDescriptor];
        id<MTLTexture> outputTexture = [device newTextureWithDescriptor:textureDescriptor];
        
        // Copy input data to texture
        MTLRegion region = MTLRegionMake2D(0, 0, imageWidth, imageHeight);
        [inputTexture replaceRegion:region
                        mipmapLevel:0
                          withBytes:inputImage.data()
                        bytesPerRow:imageWidth * channels * sizeof(float)];
        
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
        
        [computeEncoder setComputePipelineState:pipelineState];
        [computeEncoder setTexture:inputTexture atIndex:0];
        [computeEncoder setTexture:outputTexture atIndex:1];
        [computeEncoder setBuffer:kernelBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:kernelSizeBuffer offset:0 atIndex:1];
        
        // Calculate optimal thread group size
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadGroups = MTLSizeMake(
            (imageWidth + threadGroupSize.width - 1) / threadGroupSize.width,
            (imageHeight + threadGroupSize.height - 1) / threadGroupSize.height,
            1
        );
        
        [computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupSize];
        [computeEncoder endEncoding];
        
        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy output data back to CPU
        outputImage.resize(imageWidth * imageHeight * channels);
        [outputTexture getBytes:outputImage.data()
                    bytesPerRow:imageWidth * channels * sizeof(float)
                     fromRegion:region
                    mipmapLevel:0];
    }

    // Example method to demonstrate performance comparison between CPU and GPU
    void performanceTest(int imageWidth, int imageHeight) {
        const int channels = 4; // RGBA
        const int kernelSize = 5;
        
        // Create sample data
        std::vector<float> inputImage(imageWidth * imageHeight * channels, 1.0f);
        std::vector<float> outputImage(imageWidth * imageHeight * channels);
        
        // Create a simple Gaussian blur kernel
        std::vector<float> kernel = {
            1/256.0f, 4/256.0f, 6/256.0f, 4/256.0f, 1/256.0f,
            4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
            6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f, 6/256.0f,
            4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
            1/256.0f, 4/256.0f, 6/256.0f, 4/256.0f, 1/256.0f
        };
        
        // Measure GPU performance
        auto startGPU = std::chrono::high_resolution_clock::now();
        applyConvolution(inputImage, outputImage, kernel, imageWidth, imageHeight, channels, kernelSize);
        auto endGPU = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> gpuTime = endGPU - startGPU;
        std::cout << "GPU convolution time: " << gpuTime.count() << " ms" << std::endl;
        
        // Compare with CPU implementation (simple version)
        auto startCPU = std::chrono::high_resolution_clock::now();
        std::vector<float> cpuOutput(imageWidth * imageHeight * channels);
        cpuConvolution(inputImage, cpuOutput, kernel, imageWidth, imageHeight, channels, kernelSize);
        auto endCPU = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> cpuTime = endCPU - startCPU;
        std::cout << "CPU convolution time: " << cpuTime.count() << " ms" << std::endl;
        std::cout << "Speedup: " << cpuTime.count() / gpuTime.count() << "x" << std::endl;
    }
    
private:
    // Simple CPU implementation for comparison
    void cpuConvolution(
        const std::vector<float>& input,
        std::vector<float>& output,
        const std::vector<float>& kernel,
        int width,
        int height,
        int channels,
        int kernelSize
    ) {
        const int kernelRadius = kernelSize / 2;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < channels; c++) {
                    float sum = 0.0f;
                    
                    for (int ky = 0; ky < kernelSize; ky++) {
                        for (int kx = 0; kx < kernelSize; kx++) {
                            int sampleX = x + kx - kernelRadius;
                            int sampleY = y + ky - kernelRadius;
                            
                            // Clamp to edge
                            sampleX = std::max(0, std::min(width - 1, sampleX));
                            sampleY = std::max(0, std::min(height - 1, sampleY));
                            
                            float inputValue = input[(sampleY * width + sampleX) * channels + c];
                            float kernelValue = kernel[ky * kernelSize + kx];
                            
                            sum += inputValue * kernelValue;
                        }
                    }
                    
                    output[(y * width + x) * channels + c] = sum;
                }
            }
        }
    }
};

// Main function to demonstrate usage
int main() {
    try {
        MetalConvolution metalConv;
        
        // Test with different image sizes
        std::cout << "Testing with 512x512 image:" << std::endl;
        metalConv.performanceTest(512, 512);
        
        std::cout << "\nTesting with 1024x1024 image:" << std::endl;
        metalConv.performanceTest(1024, 1024);
        
        std::cout << "\nTesting with 2048x2048 image:" << std::endl;
        metalConv.performanceTest(2048, 2048);
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}