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
    if (argc != 6) {
        std::cout << "usage: ./cpu-path-tracer scene_file x_res y_res samples_per_pixel image_file\n";
        exit(0);
    }

    // Load scene
    const std::string scene_file = argv[1];
    const size_t width = std::stol(argv[2]);
    const size_t height = std::stol(argv[3]);
    const size_t samples_per_pixel = std::stol(argv[4]);
    const std::string image_file = argv[5];
    Scene scene(scene_file, width, height);

    // Run render kernel
    std::cout << "Rendering image...\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    render_scene(&scene, width, height, samples_per_pixel);
    gpuErrchk(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "\rRendered image in " << ms_int.count() << " ms\n";

    // Copy image data back to device
    Image h_image(width, height);
    gpuErrchk(cudaMemcpy(h_image.data, scene.image_data, width * height * sizeof(float3), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());

    // Save image
    std::cout << "Saving image...\n";
    t1 = std::chrono::high_resolution_clock::now();
    h_image.tone_map(EXPOSURE, GAMMA);
    h_image.save_png(image_file);
    t2 = std::chrono::high_resolution_clock::now();
    ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "\rSaved image in " << ms_int.count() << " ms\n";
}