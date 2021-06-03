cmake_minimum_required(VERSION 3.15)

project(PathTracer LANGUAGES CXX CUDA)  # Project for C++ code
project(cuda_base CXX CUDA)             # Separate project for all CUDA code

# Setup cuda_base project
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_CXX_STANDARD 14)
file(GLOB_RECURSE CUDA_SRCS CONFIGURE_DEPENDS
        "src/*.cu"
        "src/*.cuh"
        "include/*.cu"
        "include/*.cuh")
add_library(cuda_base ${CUDA_SRCS})
target_compile_features(cuda_base PUBLIC cxx_std_14)
set_target_properties( cuda_base PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30 -lineinfo")

# Setup PathTracer project
file(GLOB_RECURSE CPP_SRCS CONFIGURE_DEPENDS
        "src/*.cpp"
        "src/*.h"
        "include/*.cpp"
        "include/*.h")
add_executable(cuda-path-tracer ${CPP_SRCS})
set_target_properties(
        cuda-path-tracer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda-path-tracer cuda_base)
