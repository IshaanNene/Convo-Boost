const char* metalEffectsCode = R"(
#include <metal_stdlib>
using namespace metal;

// Basic convolution kernel that applies the provided kernel matrix
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

    // Preserve alpha if needed
    float4 original = inputTexture.read(pos);
    result.a = original.a; // Keep original alpha

    outputTexture.write(result, pos);
}

// Brightness adjustment kernel
kernel void adjustBrightness(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float& factor [[buffer(0)]],
    uint2 pos [[thread_position_in_grid]])
{
    // Check if we're within image bounds
    if (pos.x >= inputTexture.get_width() || pos.y >= inputTexture.get_height()) {
        return;
    }

    float4 color = inputTexture.read(pos);

    // Adjust brightness (RGB channels only)
    float4 result = float4(color.rgb * factor, color.a);

    // Clamp values to valid range
    result = clamp(result, 0.0f, 1.0f);

    outputTexture.write(result, pos);
}

// Contrast adjustment kernel
kernel void adjustContrast(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float& factor [[buffer(0)]],
    uint2 pos [[thread_position_in_grid]])
{
    // Check if we're within image bounds
    if (pos.x >= inputTexture.get_width() || pos.y >= inputTexture.get_height()) {
        return;
    }

    float4 color = inputTexture.read(pos);

    // Apply contrast adjustment formula (RGB channels only)
    // Formula: (color - 0.5) * factor + 0.5
    float4 result = float4((color.rgb - 0.5f) * factor + 0.5f, color.a);

    // Clamp values to valid range
    result = clamp(result, 0.0f, 1.0f);

    outputTexture.write(result, pos);
}

// Saturation adjustment kernel
kernel void adjustSaturation(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float& factor [[buffer(0)]],
    uint2 pos [[thread_position_in_grid]])
{
    // Check if we're within image bounds
    if (pos.x >= inputTexture.get_width() || pos.y >= inputTexture.get_height()) {
        return;
    }

    float4 color = inputTexture.read(pos);

    // Calculate luminance (grayscale value)
    float luminance = dot(color.rgb, float3(0.299f, 0.587f, 0.114f));

    // Mix between grayscale and original color based on saturation factor
    float4 result = float4(mix(float3(luminance), color.rgb, factor), color.a);

    // Clamp values to valid range
    result = clamp(result, 0.0f, 1.0f);

    outputTexture.write(result, pos);
}

// Edge detection kernel
kernel void edgeDetect(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    uint2 pos [[thread_position_in_grid]])
{
    const int width = inputTexture.get_width();
    const int height = inputTexture.get_height();

    // Check if we're within image bounds
    if (pos.x >= width || pos.y >= height) {
        return;
    }

    // Sample the surrounding pixels (3x3 grid)
    float4 topLeft = inputTexture.read(uint2(clamp(int(pos.x) - 1, 0, width - 1), 
                                           clamp(int(pos.y) - 1, 0, height - 1)));
    float4 top = inputTexture.read(uint2(pos.x, clamp(int(pos.y) - 1, 0, height - 1)));
    float4 topRight = inputTexture.read(uint2(clamp(int(pos.x) + 1, 0, width - 1), 
                                            clamp(int(pos.y) - 1, 0, height - 1)));

    float4 left = inputTexture.read(uint2(clamp(int(pos.x) - 1, 0, width - 1), pos.y));
    float4 center = inputTexture.read(pos);
    float4 right = inputTexture.read(uint2(clamp(int(pos.x) + 1, 0, width - 1), pos.y));

    float4 bottomLeft = inputTexture.read(uint2(clamp(int(pos.x) - 1, 0, width - 1), 
                                              clamp(int(pos.y) + 1, 0, height - 1)));
    float4 bottom = inputTexture.read(uint2(pos.x, clamp(int(pos.y) + 1, 0, height - 1)));
    float4 bottomRight = inputTexture.read(uint2(clamp(int(pos.x) + 1, 0, width - 1), 
                                               clamp(int(pos.y) + 1, 0, height - 1)));

    // Apply Sobel operator (edge detection)
    // Horizontal gradient
    float4 gx = -topLeft - 2.0f * left - bottomLeft + topRight + 2.0f * right + bottomRight;

    // Vertical gradient
    float4 gy = -topLeft - 2.0f * top - topRight + bottomLeft + 2.0f * bottom + bottomRight;

    // Approximate magnitude
    float4 mag = sqrt(gx * gx + gy * gy);

    // Keep original alpha
    mag.a = center.a;

    outputTexture.write(mag, pos);
}

// Sepia tone effect
kernel void sepiaTone(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant float& intensity [[buffer(0)]],
    uint2 pos [[thread_position_in_grid]])
{
    // Check if we're within image bounds
    if (pos.x >= inputTexture.get_width() || pos.y >= inputTexture.get_height()) {
        return;
    }

    float4 color = inputTexture.read(pos);

    // Apply sepia transformation
    float4 sepia = float4(
        dot(color.rgb, float3(0.393f, 0.769f, 0.189f)),
        dot(color.rgb, float3(0.349f, 0.686f, 0.168f)),
        dot(color.rgb, float3(0.272f, 0.534f, 0.131f)),
        color.a
    );

    // Mix with original based on intensity
    float4 result = mix(color, sepia, intensity);

    outputTexture.write(result, pos);
}

// Pixelate effect
kernel void pixelate(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant int& blockSize [[buffer(0)]],
    uint2 pos [[thread_position_in_grid]])
{
    const int width = inputTexture.get_width();
    const int height = inputTexture.get_height();

    // Check if we're within image bounds
    if (pos.x >= width || pos.y >= height) {
        return;
    }

    // Calculate the top-left position of the block this pixel belongs to
    uint2 blockStart = uint2((pos.x / blockSize) * blockSize, (pos.y / blockSize) * blockSize);

    // Sample the color at the center of the block
    uint2 samplePos = uint2(
        clamp(blockStart.x + blockSize / 2, 0u, uint(width - 1)),
        clamp(blockStart.y + blockSize / 2, 0u, uint(height - 1))
    );

    float4 blockColor = inputTexture.read(samplePos);

    outputTexture.write(blockColor, pos);
})";
