// random_image_generator.cpp
// Utility to generate various random test images for convolution operations

#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

class ImageGenerator {
private:
    std::mt19937 rng;

public:
    ImageGenerator(unsigned int seed = 42) : rng(seed) {}

    // Generate a random noise image
    std::vector<float> generateNoiseImage(int width, int height, int channels = 4) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> image(width * height * channels);
        
        for (size_t i = 0; i < image.size(); ++i) {
            image[i] = dist(rng);
        }
        
        return image;
    }
    
    // Generate a gradient image
    std::vector<float> generateGradientImage(int width, int height, int channels = 4) {
        std::vector<float> image(width * height * channels);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float normalizedX = static_cast<float>(x) / width;
                float normalizedY = static_cast<float>(y) / height;
                
                for (int c = 0; c < channels; ++c) {
                    // Different gradient for each channel
                    if (c == 0) {
                        // Red channel: horizontal gradient
                        image[(y * width + x) * channels + c] = normalizedX;
                    } else if (c == 1) {
                        // Green channel: vertical gradient
                        image[(y * width + x) * channels + c] = normalizedY;
                    } else if (c == 2) {
                        // Blue channel: diagonal gradient
                        image[(y * width + x) * channels + c] = (normalizedX + normalizedY) / 2.0f;
                    } else {
                        // Alpha or additional channels: full opacity
                        image[(y * width + x) * channels + c] = 1.0f;
                    }
                }
            }
        }
        
        return image;
    }
    
    // Generate a checkerboard image
    std::vector<float> generateCheckerboardImage(int width, int height, int channels = 4, int squareSize = 32) {
        std::vector<float> image(width * height * channels);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                bool isWhite = ((x / squareSize) + (y / squareSize)) % 2 == 0;
                float value = isWhite ? 1.0f : 0.0f;
                
                for (int c = 0; c < channels; ++c) {
                    if (c < 3) {
                        // RGB channels
                        image[(y * width + x) * channels + c] = value;
                    } else {
                        // Alpha or additional channels: full opacity
                        image[(y * width + x) * channels + c] = 1.0f;
                    }
                }
            }
        }
        
        return image;
    }
    
    // Generate a circular pattern
    std::vector<float> generateCircularPattern(int width, int height, int channels = 4) {
        std::vector<float> image(width * height * channels);
        float centerX = width / 2.0f;
        float centerY = height / 2.0f;
        float maxDist = std::min(width, height) / 2.0f;
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float dx = x - centerX;
                float dy = y - centerY;
                float distance = std::sqrt(dx * dx + dy * dy);
                float normalizedDist = std::min(distance / maxDist, 1.0f);
                
                for (int c = 0; c < channels; ++c) {
                    if (c == 0) {
                        // Red channel: radial gradient
                        image[(y * width + x) * channels + c] = normalizedDist;
                    } else if (c == 1) {
                        // Green channel: inverse radial gradient
                        image[(y * width + x) * channels + c] = 1.0f - normalizedDist;
                    } else if (c == 2) {
                        // Blue channel: concentric circles
                        image[(y * width + x) * channels + c] = (std::sin(normalizedDist * 20) + 1.0f) / 2.0f;
                    } else {
                        // Alpha or additional channels: full opacity
                        image[(y * width + x) * channels + c] = 1.0f;
                    }
                }
            }
        }
        
        return image;
    }
    
    // Generate a test card with various patterns
    std::vector<float> generateTestCard(int width, int height, int channels = 4) {
        std::vector<float> image(width * height * channels, 0.5f);
        int sections = 4;
        int sectionWidth = width / sections;
        int sectionHeight = height / sections;
        
        for (int sy = 0; sy < sections; ++sy) {
            for (int sx = 0; sx < sections; ++sx) {
                int startX = sx * sectionWidth;
                int startY = sy * sectionHeight;
                int endX = startX + sectionWidth;
                int endY = startY + sectionHeight;
                
                // Different pattern for each section
                int patternType = (sy * sections + sx) % 5;
                
                for (int y = startY; y < endY; ++y) {
                    for (int x = startX; x < endX; ++x) {
                        if (x >= width || y >= height) continue;
                        
                        float value = 0.5f;
                        
                        switch (patternType) {
                            case 0: // Solid color
                                value = 0.8f;
                                break;
                            case 1: // Gradient
                                value = static_cast<float>(x - startX) / sectionWidth;
                                break;
                            case 2: // Checkerboard
                                value = ((x / 16) + (y / 16)) % 2 == 0 ? 1.0f : 0.0f;
                                break;
                            case 3: // Stripes
                                value = (x % 16) < 8 ? 1.0f : 0.0f;
                                break;
                            case 4: // Dots
                                {
                                    int cx = (x - startX) % 32 - 16;
                                    int cy = (y - startY) % 32 - 16;
                                    value = (cx * cx + cy * cy < 64) ? 1.0f : 0.0f;
                                }
                                break;
                        }
                        
                        for (int c = 0; c < channels; ++c) {
                            if (c < 3) {
                                // Channel-specific values
                                if (c == sx % 3) {
                                    image[(y * width + x) * channels + c] = value;
                                } else {
                                    image[(y * width + x) * channels + c] = 0.2f;
                                }
                            } else {
                                // Alpha or additional channels
                                image[(y * width + x) * channels + c] = 1.0f;
                            }
                        }
                    }
                }
            }
        }
        
        return image;
    }
    
    // Save image to a PPM file (RGB only)
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
};

int main() {
    const int width = 512;
    const int height = 512;
    const int channels = 4; // RGBA
    
    ImageGenerator generator;
    
    // Generate different types of images
    auto noiseImage = generator.generateNoiseImage(width, height, channels);
    auto gradientImage = generator.generateGradientImage(width, height, channels);
    auto checkerboardImage = generator.generateCheckerboardImage(width, height, channels);
    auto circularImage = generator.generateCircularPattern(width, height, channels);
    auto testCardImage = generator.generateTestCard(width, height, channels);
    
    // Save images to files
    generator.saveImageToPPM("noise.ppm", noiseImage, width, height, channels);
    generator.saveImageToPPM("gradient.ppm", gradientImage, width, height, channels);
    generator.saveImageToPPM("checkerboard.ppm", checkerboardImage, width, height, channels);
    generator.saveImageToPPM("circular.ppm", circularImage, width, height, channels);
    generator.saveImageToPPM("testcard.ppm", testCardImage, width, height, channels);
    
    std::cout << "Generated 5 test images (saved as PPM files)" << std::endl;
    
    return 0;
}