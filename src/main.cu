#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include "camera.cuh"
#include "image.h"
#include "render.cuh"
#include "scene.h"

#define EXPOSURE            2.0f    // Used for tone mapping
#define GAMMA               2.2f    // Used for gamma correction

void load_obj(const std::string &filename, std::vector<Triangle> &triangles)
{
    std::cout << "Loading .obj file...\n";
    std::ifstream obj_file (filename);
    std::vector<float3> vertices;
    vertices.emplace_back();
    size_t count = 0;
    std::string line;

    auto t1 = std::chrono::high_resolution_clock::now();
    while (!obj_file.eof()) {
        getline(obj_file, line);
        if (obj_file.eof()) { break; }

        char type;
        std::istringstream iss(line);
        iss >> type;

        if (type == 'v') {
            float3 v;
            iss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        } else if (type == 'f') {
            int i1, i2, i3;
            iss >> i1 >> i2 >> i3;
            i1 = (i1 < 0) ? i1 + vertices.size() : i1;
            i2 = (i2 < 0) ? i2 + vertices.size() : i2;
            i3 = (i3 < 0) ? i3 + vertices.size() : i3;
            triangles.emplace_back(vertices[i1], vertices[i2], vertices[i3]);
            count++;
        } else {
            std::cerr << "A parsing error occurred while reading in "
                      << "the .obj file" << filename << "\n";
            return;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Loaded " << count << " triangles in " << ms_int.count() << " ms\n";
}

int main(int argc, char **argv)
{
    const size_t width = 1920;
    const size_t height = 1080;
    const size_t samples_per_pixel = 512;

    // Load Scene File
    Scene h_scene("../scene/balls.scene", width, height);

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