cmake_minimum_required(VERSION 3.15)

project(PathTracer LANGUAGES CXX CUDA)

# Setup CUDA project
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_CUDA_STANDARD 14)

# Include source files and compile executable
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        "src/*"
        "include/*")
add_executable(cuda-path-tracer ${SRC_FILES})
set_target_properties(cuda-path-tracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES 50)