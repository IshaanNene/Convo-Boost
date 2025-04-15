#!/bin/bash

# Cyberpunk color palette
NEON_PINK='\033[38;5;198m'
NEON_BLUE='\033[38;5;51m'
NEON_GREEN='\033[38;5;46m'
NEON_YELLOW='\033[38;5;226m'
NEON_PURPLE='\033[38;5;165m'
NEON_CYAN='\033[38;5;87m'
NEON_RED='\033[38;5;196m'
NEON_ORANGE='\033[38;5;214m'

# Base colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Styles
BOLD='\033[1m'
DIM='\033[2m'
BLINK='\033[5m'
REVERSE='\033[7m'
UNDERLINE='\033[4m'

# Get terminal width
TERM_WIDTH=$(tput cols)

# Get the absolute path of the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to create a progress bar
progress_bar() {
    local current=$1
    local total=$2
    local width=40
    local percentage=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))
    
    printf "${NEON_BLUE}[${NEON_CYAN}"
    for ((i = 0; i < filled; i++)); do printf "█"; done
    printf "${DIM}"
    for ((i = 0; i < empty; i++)); do printf "░"; done
    printf "${NEON_BLUE}]${NEON_GREEN} %3d%%${NC}" $percentage
}

# Function to print centered text
print_centered() {
    local text="$1"
    local color="$2"
    local padding=$(( (TERM_WIDTH - ${#text}) / 2 ))
    printf "%${padding}s" ''
    echo -e "${color}${text}${NC}"
}

# Function to print a neon box
print_neon_box() {
    local text="$1"
    local color="$2"
    local text_length=${#text}
    local box_width=$((text_length + 4))
    local padding=$(( (TERM_WIDTH - box_width) / 2 ))
    
    printf "%${padding}s" ''
    echo -e "${color}╔═${BOLD}${text}${NC}${color}═╗${NC}"
    
    printf "%${padding}s" ''
    echo -e "${color}║ ${NC}${BOLD}${text}${NC}${color} ║${NC}"
    
    printf "%${padding}s" ''
    echo -e "${color}╚$(printf '═%.0s' $(seq 1 $((box_width - 2))))╝${NC}"
}

# Function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${NEON_GREEN}${BOLD}[${BLINK}✓${NC}${NEON_GREEN}${BOLD}] ${NEON_CYAN}$2 ${NEON_GREEN}[SUCCESS]${NC}"
    else
        echo -e "${NEON_RED}${BOLD}[${BLINK}✗${NC}${NEON_RED}${BOLD}] ${NEON_PINK}$2 ${NEON_RED}[FAILED]${NC}"
    fi
}

# Function to print status messages
print_status() {
    echo -e "${NEON_BLUE}${BOLD}[${BLINK}*${NC}${NEON_BLUE}${BOLD}] ${NEON_CYAN}$1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${NEON_YELLOW}${BOLD}[${BLINK}!${NC}${NEON_YELLOW}${BOLD}] ${NEON_ORANGE}$1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${NEON_RED}${BOLD}[${BLINK}✗${NC}${NEON_RED}${BOLD}] ${NEON_PINK}$1${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${NEON_GREEN}${BOLD}[${BLINK}✓${NC}${NEON_GREEN}${BOLD}] ${NEON_CYAN}$1${NC}"
}

# Function to print cyberpunk banner
print_banner() {
    clear
    echo
    print_centered "╔══════════════════════════════════════════════════════════╗" "${NEON_BLUE}"
    print_centered "║              CONVO-BOOST IMAGE PROCESSING TEST           ║" "${NEON_BLUE}"
    print_centered "╚══════════════════════════════════════════════════════════╝" "${NEON_BLUE}"
    echo
    print_centered " ██████╗ ██████╗ ███╗   ██╗██╗   ██╗ ██████╗ " "${NEON_CYAN}"
    print_centered "██╔════╝██╔═══██╗████╗  ██║██║   ██║██╔═══██╗" "${NEON_PURPLE}"
    print_centered "██║     ██║   ██║██╔██╗ ██║██║   ██║██║   ██║" "${NEON_PINK}"
    print_centered "██║     ██║   ██║██║╚██╗██║██║   ██║██║   ██║" "${NEON_BLUE}"
    print_centered "╚██████╗╚██████╔╝██║ ╚████║╚██████╔╝╚██████╔╝" "${NEON_GREEN}"
    print_centered " ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ " "${NEON_YELLOW}"
    echo
    print_centered "██████╗  ██████╗  ██████╗ ███████╗████████╗" "${NEON_CYAN}"
    print_centered "██╔══██╗██╔═══██╗██╔═══██╗██╔════╝╚══██╔══╝" "${NEON_PURPLE}"
    print_centered "██████╔╝██║   ██║██║   ██║███████╗   ██║   " "${NEON_PINK}"
    print_centered "██╔══██╗██║   ██║██║   ██║╚════██║   ██║   " "${NEON_BLUE}"
    print_centered "██████╔╝╚██████╔╝╚██████╔╝███████║   ██║   " "${NEON_GREEN}"
    print_centered "╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝   " "${NEON_YELLOW}"
    echo
    print_centered "INITIALIZING IMAGE PROCESSING TEST SEQUENCE..." "${NEON_CYAN}"
    echo
}

# Function to print test header
print_test_header() {
    echo
    print_neon_box " TEST: $1 " "${NEON_PURPLE}"
    echo
}

# Function to print section divider
print_divider() {
    echo
    print_centered "════════════════════════════════════════" "${NEON_BLUE}"
    echo
}

# Function to simulate loading animation
simulate_loading() {
    local text="$1"
    local duration=$2
    local interval=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local temp
    
    for ((i = 0; i < duration; i++)); do
        temp=${spinstr#?}
        printf "${NEON_CYAN}${BOLD}[%c]${NC} ${NEON_BLUE}${text}${NC}" "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $interval
        printf "\r"
    done
    printf "\n"
}

# Function to check if build directory exists
check_build() {
    if [ ! -d "build" ]; then
        print_status "Creating build directory..."
        mkdir -p build
    fi
}

# Function to build the project
build_project() {
    print_status "Building project..."
    cd build || return 1
    cmake .. && make -j$(sysctl -n hw.ncpu)
    BUILD_RESULT=$?
    cd ..
    return $BUILD_RESULT
}

# Function to test random image generator
test_random_image_generator() {
    print_test_header "RANDOM IMAGE GENERATOR"
    print_status "Generating test images..."
    ./build/random_image_generator
    GENERATOR_RESULT=$?
    print_result $GENERATOR_RESULT "Random image generation"
    
    # Check if images were created
    if [ -f "noise.ppm" ] && [ -f "gradient.ppm" ] && [ -f "checkerboard.ppm" ] && [ -f "circular.ppm" ] && [ -f "testcard.ppm" ]; then
        print_success "All test images generated successfully"
        return 0
    else
        print_error "Some test images were not generated"
        return 1
    fi
}

# Function to test image effects processor
test_image_effects() {
    print_test_header "IMAGE EFFECTS PROCESSOR"
    print_status "Processing test images..."
    ./build/image_effects_processor
    EFFECTS_RESULT=$?
    print_result $EFFECTS_RESULT "Image effects processing"
    
    # Check if effect images were created
    if [ -f "effect_blur.ppm" ] && [ -f "effect_brightness.ppm" ] && [ -f "effect_contrast.ppm" ] && [ -f "effect_saturation.ppm" ]; then
        print_success "All effect images generated successfully"
        return 0
    else
        print_error "Some effect images were not generated"
        return 1
    fi
}

# Function to test metal convolution
test_metal_convolution() {
    print_test_header "METAL CONVOLUTION"
    print_status "Testing convolution performance..."
    ./build/metal_convolution
    CONVOLUTION_RESULT=$?
    print_result $CONVOLUTION_RESULT "Metal convolution performance test"
    return $CONVOLUTION_RESULT
}

# Function to test PPM to PNG conversion
test_ppm_to_png() {
    print_test_header "PPM TO PNG CONVERSION"
    print_status "Converting PPM images to PNG..."
    ./build/ppm_to_png .
    CONVERSION_RESULT=$?
    print_result $CONVERSION_RESULT "PPM to PNG conversion"
    
    # Check if PNG files were created
    if [ -f "noise.png" ] && [ -f "gradient.png" ] && [ -f "checkerboard.png" ]; then
        print_success "All images converted successfully"
        return 0
    else
        print_error "Some images were not converted"
        return 1
    fi
}

# Main test sequence
print_banner

# Initialize test environment
print_neon_box " INITIALIZING TEST ENVIRONMENT " "${NEON_BLUE}"
echo

# Check and create build directory
check_build

# Build the project
print_status "Building project..."
build_project
BUILD_RESULT=$?
print_result $BUILD_RESULT "Project build"

if [ $BUILD_RESULT -ne 0 ]; then
    print_error "Build failed. Exiting..."
    exit 1
fi

# Run tests
print_neon_box " RUNNING TESTS " "${NEON_PURPLE}"
echo

# Test 1: Random Image Generator
test_random_image_generator
GENERATOR_RESULT=$?

# Test 2: Image Effects Processor
test_image_effects
EFFECTS_RESULT=$?

# Test 3: Metal Convolution
test_metal_convolution
CONVOLUTION_RESULT=$?

# Test 4: PPM to PNG Conversion
test_ppm_to_png
CONVERSION_RESULT=$?

# Print final summary
echo
print_neon_box " TEST SUMMARY " "${NEON_PURPLE}"
echo
print_result $BUILD_RESULT "Project build"
print_result $GENERATOR_RESULT "Random image generator"
print_result $EFFECTS_RESULT "Image effects processor"
print_result $CONVOLUTION_RESULT "Metal convolution"
print_result $CONVERSION_RESULT "PPM to PNG conversion"

# Cleanup
print_neon_box " CLEANUP SEQUENCE " "${NEON_RED}"
echo
simulate_loading "Cleaning up temporary files" 5
rm -f *.ppm *.png
print_centered "TEST SEQUENCE COMPLETED" "${NEON_GREEN}"
echo 