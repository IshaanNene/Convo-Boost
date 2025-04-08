#include <iostream>
#include <string>
#include <filesystem>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

bool convert_ppm_to_png(const std::string& ppm_path, const std::string& png_path) {
    // Read PPM file
    int width, height, channels;
    unsigned char* data = stbi_load(ppm_path.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load PPM image: " << ppm_path << std::endl;
        return false;
    }

    // Write PNG file
    int success = stbi_write_png(png_path.c_str(), width, height, channels, data, width * channels);
    stbi_image_free(data);

    if (!success) {
        std::cerr << "Failed to write PNG image: " << png_path << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        std::cout << "This will convert all PPM files in the specified directory to PNG format." << std::endl;
        return 1;
    }

    std::string dir_path = argv[1];
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        std::cerr << "Error: " << dir_path << " is not a valid directory" << std::endl;
        return 1;
    }

    int converted = 0;
    int failed = 0;

    // Convert all PPM files in the directory
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".ppm") {
            std::string ppm_path = entry.path().string();
            fs::path png_path = entry.path();
            png_path.replace_extension(".png");
            
            std::cout << "Converting " << ppm_path << " to " << png_path << "... ";
            
            if (convert_ppm_to_png(ppm_path, png_path.string())) {
                std::cout << "Success" << std::endl;
                converted++;
            } else {
                std::cout << "Failed" << std::endl;
                failed++;
            }
        }
    }

    std::cout << "\nConversion complete!" << std::endl;
    std::cout << "Successfully converted: " << converted << " files" << std::endl;
    std::cout << "Failed conversions: " << failed << " files" << std::endl;

    return 0;
} 