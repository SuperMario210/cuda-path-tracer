#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include "camera.cuh"
#include "image.h"
#include "render.cuh"

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
    const size_t samples_per_pixel = 64;

    // Setup BVH
    std::vector<Triangle> triangles;
    load_obj("../obj/dragon.obj", triangles);
//    load_obj("../obj/bunny_lowpoly.obj", triangles);
    BVH bvh_h(triangles);
    BVH *bvh_d;
    gpuErrchk(cudaMalloc(&bvh_d, sizeof(BVH)));
    gpuErrchk(cudaMemcpy(bvh_d, &bvh_h, sizeof(BVH), cudaMemcpyHostToDevice));

    // Setup environment map
    const EnvironmentMap envmap_h("../background/studio.hdr");
    EnvironmentMap *envmap_d;
    gpuErrchk(cudaMalloc(&envmap_d, sizeof(EnvironmentMap)));
    gpuErrchk(cudaMemcpy(envmap_d, &envmap_h, sizeof(EnvironmentMap), cudaMemcpyHostToDevice));

    // Setup camera
    float3 origin = make_float3(-1.1, 0.2, -0.8);
    float3 look_at = make_float3(0, 0, 0);
    float fov = 38;
    float aspect_ratio = float(width) / float(height);
    float aperture = 0.01;
    float focus_dist = 1.15; // length(look_at - origin);

//    float3 origin = make_float3(0, 1, 5);
//    float3 look_at = make_float3(0, 0.5, 0);
//    float fov = 38;
//    float aspect_ratio = float(width) / float(height);
//    float aperture = 0.0;
//    float focus_dist = length(look_at - origin);

    Camera camera_h(origin, look_at, fov, aspect_ratio, aperture, focus_dist);
    Camera *camera_d;
    gpuErrchk(cudaMalloc(&camera_d, sizeof(Camera)));
    gpuErrchk(cudaMemcpy(camera_d, &camera_h, sizeof(Camera), cudaMemcpyHostToDevice));

    // Setup image
    Image image_h(width, height);
    float3 *image_d;
    gpuErrchk(cudaMalloc(&image_d, width * height * sizeof(float3)));
    PathData *paths;
    gpuErrchk(cudaMalloc(&paths, sizeof(PathData)));

    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    // Run render kernel
    std::cout << "Rendering image...\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    launch_render_kernel(bvh_d, envmap_d, camera_d, image_d, width, height, samples_per_pixel, grid, block, paths);
    gpuErrchk(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "\rRendered image in " << ms_int.count() << " ms\n";

    // Copy image data back to device
    gpuErrchk(cudaMemcpy(image_h.data, image_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(bvh_d));
    gpuErrchk(cudaFree(envmap_d));
    gpuErrchk(cudaFree(camera_d));
    gpuErrchk(cudaFree(image_d));
    gpuErrchk(cudaDeviceSynchronize());

    // Save image
    std::cout << "Saving image...\n";
    t1 = std::chrono::high_resolution_clock::now();
    image_h.tone_map(EXPOSURE, GAMMA);
    image_h.save_png("../img/test");
    t2 = std::chrono::high_resolution_clock::now();
    ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "\rSaved image in " << ms_int.count() << " ms\n";
}