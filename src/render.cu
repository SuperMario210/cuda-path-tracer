#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void render_kernel(float3 *image_data, size_t width, size_t height) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = (height - y - 1) * width + x;

    image_data[i] = make_float3(float(x) / float(width), float(y) / float(height), 1);
}

void launch_render_kernel(float3 *image_data, size_t width, size_t height, dim3 grid, dim3 block) {
    render_kernel <<< grid, block >>>(image_data, width, height);
}