#include <cuda_runtime.h>
#include <vector_types.h>
#include "camera.cuh"
#include "object.cuh"
#include <curand_kernel.h>

__device__ Ray scatter(const Ray &in_ray, const Intersection &intersect, float3 &attenuation, curandState &rand_state)
{
    float z = curand_uniform(&rand_state) * 2.0f - 1.0f;
    float a = curand_uniform(&rand_state) * 2.0f * PI;
    float r = sqrtf(1.0f - z * z);
    float x = r * cosf(a);
    float y = r * sinf(a);
    float3 p = make_float3(x, y, z);

    attenuation = make_float3(0.5);
    return {intersect.position, intersect.normal + p};
}

__device__ float3 path_trace(Ray r, curandState &rand_state)
{
    Plane plane(make_float3(0), make_float3(0, 1, 0));
    Sphere sphere(make_float3(0, 0.5, 2), 0.5);

    float3 color = make_float3(1);
    for (int depth = 0; depth < 64; depth++) {
        // Intersect scene
        Intersection intersect;
        bool hit1 = plane.intersect(r, intersect);
        bool hit2 = sphere.intersect(r, intersect);

        if (hit1 || hit2) {
            float3 attenuation;
            r = scatter(r, intersect, attenuation, rand_state);
            color *= attenuation;
        } else {
            float3 unit_direction = normalize(r.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            float3 c = (1.0f - t) * make_float3(1.0, 1.0, 1.0) + t * make_float3(0.5, 0.7, 1.0);
            color *= c;
            break;
        }
    }

    return color;
}

__global__ void render_kernel(Camera *camera, float3 *image_data, size_t width, size_t height,
                              size_t samples_per_pixel)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = (height - y - 1) * width + x;

    curandState rand_state;
    curand_init(1984 + i, 0, 0, &rand_state);

    float3 accum_color = make_float3(0);
    for (int samp = 0; samp < samples_per_pixel; samp++) {
        float u = curand_uniform(&rand_state) - 0.5f;
        float v = curand_uniform(&rand_state) - 0.5f;
        Ray ray = camera->cast_ray(float(x + v) / float(width), float(y + u) / float(height), rand_state);
        accum_color += path_trace(ray, rand_state);
    }

    image_data[i] = accum_color / samples_per_pixel;
}

__host__ void launch_render_kernel(Camera *camera, float3 *image_data, size_t width, size_t height,
                                   size_t samples_per_pixel, dim3 grid, dim3 block)
                          {
    render_kernel <<< grid, block >>>(camera, image_data, width, height, samples_per_pixel);
}