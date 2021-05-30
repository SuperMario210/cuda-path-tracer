#include <iostream>
#include <chrono>
#include "camera.cuh"
#include "render.cuh"

#define EXPOSURE            1.0f    // Used for tone mapping
#define GAMMA               2.2f    // Used for gamma correction

int main(int argc, char **argv)
{
    const size_t width = 1920/2;
    const size_t height = 1080/2;
    const size_t samples_per_pixel = 1024/4;

    // Setup environment map
    const EnvironmentMap envmap("../background/studio.hdr");
    EnvironmentMapData envmap_h = envmap.get_data();
    EnvironmentMapData *envmap_d;
    gpuErrchk(cudaMalloc(&envmap_d, sizeof(EnvironmentMapData)));
    gpuErrchk(cudaMemcpy(envmap_d, &envmap_h, sizeof(EnvironmentMapData), cudaMemcpyHostToDevice));

    // Setup camera
    float3 origin = make_float3(0, 1, 0);
    float3 look_at = make_float3(0, 0.5, 2);
    float fov = 60;
    float aspect_ratio = float(width) / float(height);
    float aperture = 0;
    float focus_dist = length(look_at - origin);
    Camera camera_h(origin, look_at, fov, aspect_ratio, aperture, focus_dist);
    Camera *camera_d;
    gpuErrchk(cudaMalloc(&camera_d, sizeof(Camera)));
    gpuErrchk(cudaMemcpy(camera_d, &camera_h, sizeof(Camera), cudaMemcpyHostToDevice));

    // Setup image
    Image image_h(width, height);
    float3 *image_d;
    gpuErrchk(cudaMalloc(&image_d, width * height * sizeof(float3)));
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    // Run render kernel
    std::cout << "Rendering image...\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    launch_render_kernel(envmap_d, camera_d, image_d, width, height, samples_per_pixel, grid, block);
    gpuErrchk(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "\rRendered image in " << ms_int.count() << " ms\n";

    // Copy image data back to device
    gpuErrchk(cudaMemcpy(image_h.data, image_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost));
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