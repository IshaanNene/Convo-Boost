# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ishaannene/Desktop/HP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ishaannene/Desktop/HP/build

# Include any dependencies generated for this target.
include CMakeFiles/image_effects_processor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/image_effects_processor.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/image_effects_processor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/image_effects_processor.dir/flags.make

CMakeFiles/image_effects_processor.dir/codegen:
.PHONY : CMakeFiles/image_effects_processor.dir/codegen

CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.o: CMakeFiles/image_effects_processor.dir/flags.make
CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.o: /Users/ishaannene/Desktop/HP/image_effects_processor.mm
CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.o: CMakeFiles/image_effects_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ishaannene/Desktop/HP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.o -MF CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.o.d -o CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.o -c /Users/ishaannene/Desktop/HP/image_effects_processor.mm

CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ishaannene/Desktop/HP/image_effects_processor.mm > CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.i

CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ishaannene/Desktop/HP/image_effects_processor.mm -o CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.s

# Object files for target image_effects_processor
image_effects_processor_OBJECTS = \
"CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.o"

# External object files for target image_effects_processor
image_effects_processor_EXTERNAL_OBJECTS =

image_effects_processor: CMakeFiles/image_effects_processor.dir/image_effects_processor.mm.o
image_effects_processor: CMakeFiles/image_effects_processor.dir/build.make
image_effects_processor: CMakeFiles/image_effects_processor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ishaannene/Desktop/HP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable image_effects_processor"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/image_effects_processor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/image_effects_processor.dir/build: image_effects_processor
.PHONY : CMakeFiles/image_effects_processor.dir/build

CMakeFiles/image_effects_processor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/image_effects_processor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/image_effects_processor.dir/clean

CMakeFiles/image_effects_processor.dir/depend:
	cd /Users/ishaannene/Desktop/HP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ishaannene/Desktop/HP /Users/ishaannene/Desktop/HP /Users/ishaannene/Desktop/HP/build /Users/ishaannene/Desktop/HP/build /Users/ishaannene/Desktop/HP/build/CMakeFiles/image_effects_processor.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/image_effects_processor.dir/depend

