#include <iostream>
#include <chrono>
#include "render.cuh"
#include "image.h"

#define EXPOSURE            2.0f    // Used for tone mapping
#define GAMMA               2.2f    // Used for gamma correction

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

int main(int argc, char **argv)
{
    size_t width = 1920;
    size_t height = 1080;

    // Allocate memory for image on host and device
    Image image(width, height);
    float3* image_d;
    gpuErrchk(cudaMalloc(&image_d, width * height * sizeof(float3)));
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    // Run render kernel
    std::cout << "Rendering image...\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    launch_render_kernel(image_d, width, height, grid, block);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "\rRendered image in " << ms_int.count() << " ms\n";

    // Copy image data back to device
    gpuErrchk(cudaMemcpy(image.data, image_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(image_d));

    // Save image
    image.tone_map(EXPOSURE, GAMMA);
    image.save_png("../img/test");
}