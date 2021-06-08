#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include "camera.cuh"
#include "image.h"
#include "render.cuh"
#include "scene.cuh"

#define EXPOSURE            2.0f    // Used for tone mapping
#define GAMMA               2.2f    // Used for gamma correction

int main(int argc, char **argv)
{
    const size_t width = 1920;
    const size_t height = 1080;
    const size_t samples_per_pixel = 512;

    // Load Scene File
    Scene h_scene("../scene/dragon.scene", width, height);

    // Run render kernel
    std::cout << "Rendering image...\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    launch_render_kernel(&h_scene, width, height, samples_per_pixel);
    gpuErrchk(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "\rRendered image in " << ms_int.count() << " ms\n";

    // Copy image data back to device
    Image h_image(width, height);
    gpuErrchk(cudaMemcpy(h_image.data, h_scene.image_data, width * height * sizeof(float3), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());

    // Save image
    std::cout << "Saving image...\n";
    t1 = std::chrono::high_resolution_clock::now();
    h_image.tone_map(EXPOSURE, GAMMA);
    h_image.save_png("../img/test");
    t2 = std::chrono::high_resolution_clock::now();
    ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "\rSaved image in " << ms_int.count() << " ms\n";
}