#include <cuda_runtime.h>
#include <vector_types.h>
#include "camera.cuh"
#include "object.cuh"
#include "environment_map.h"
#include <curand_kernel.h>

#define IMPORTANCE_SAMPLING

__device__ Ray scatter(const Ray &in_ray, const Intersection &intersect, float3 &attenuation, curandState &rand_state)
{
    float z = curand_uniform(&rand_state) * 2.0f - 1.0f;
    float a = curand_uniform(&rand_state) * 2.0f * PI;
    float r = sqrtf(1.0f - z * z);
    float x = r * cosf(a);
    float y = r * sinf(a);
    float3 dir = make_float3(x, y, z) + intersect.normal;

    if (dir.x < EPSILON && dir.y < EPSILON && dir.z < EPSILON)
        dir = intersect.normal;

    attenuation = make_float3(0.5);
    return {intersect.position, dir};
}

__device__ float3 sample_envmap(const EnvironmentMapData *envmap, const float3 &direction)
{
    auto dir = normalize(direction);
    auto phi = atan2f(dir.z, dir.x) * (0.5f / PI) + 0.5f;
    auto theta = acosf(dir.y) / PI;
    return make_float3(tex2D<float4>(envmap->texture_obj, phi, theta));
}

__device__ float3 sample_lights(const EnvironmentMapData *envmap, curandState &rand_state)
{
    size_t x = envmap->marginal_lookup[(size_t)(curand_uniform(&rand_state) * envmap->width - 0.5)];
    size_t y = envmap->conditional_lookup[x * envmap->height + (size_t)(curand_uniform(&rand_state) * envmap->height - 0.5)];

    auto phi = 2.0f * x * PI / envmap->width;
    auto theta = y * PI / envmap->height;
    auto sin_t = -sinf(theta);

    return make_float3(sin_t * cosf(phi), -cosf(theta), sin_t * sinf(phi));
}

__device__ float envmap_pdf(const EnvironmentMapData *envmap, float3 &direction)
{
    auto dir = normalize(direction);
    auto phi = atan2f(dir.z, dir.x) * (0.5f / PI) + 0.5f;
    auto theta = acosf(dir.y) / PI;
    return tex2D<float4>(envmap->texture_obj, phi, theta).w;
}

__device__ float diffuse_pdf(const Ray &in_ray, const Intersection &intersect, const Ray &sample)
{
    float cosine = dot(intersect.normal, normalize(sample.direction));
    return cosine < 0 ? 0 : (cosine / PI);
}

__device__ float3 path_trace(Ray r, curandState &rand_state, EnvironmentMapData *envmap)
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

#ifdef IMPORTANCE_SAMPLING

            float3 attenuation;
            Ray r_new(make_float3(0), make_float3(0));

            // TODO: figure out how to reduce divergence caused by this branch
            if (curand_uniform(&rand_state) < 0.5) {
                r_new = scatter(r, intersect, attenuation, rand_state);
            } else {
                r_new.direction = sample_lights(envmap, rand_state);
                r_new.origin = intersect.position;
                attenuation = make_float3(0.5);
            }

            float env_pdf = envmap_pdf(envmap, r_new.direction);
            float diff_pdf = diffuse_pdf(r, intersect, r_new);
            float mixed_pdf = (env_pdf + diff_pdf) / 2;

            color *= attenuation / mixed_pdf * diff_pdf;
            r = r_new;

#else

            float3 attenuation;
            r = scatter(r, intersect, attenuation, rand_state);
            color *= attenuation;

#endif

        } else {
            color = color * sample_envmap(envmap, r.direction);
            return color;
        }
    }

    return make_float3(0);
}

__global__ void render_kernel(EnvironmentMapData *envmap, Camera *camera, float3 *image_data, size_t width, size_t height,
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
        accum_color += path_trace(ray, rand_state, envmap);
    }

    image_data[i] = accum_color / samples_per_pixel;
}

__host__ void launch_render_kernel(EnvironmentMapData *envmap, Camera *camera, float3 *image_data, size_t width, size_t height,
                                   size_t samples_per_pixel, dim3 grid, dim3 block)
                          {
    render_kernel <<< grid, block >>>(envmap, camera, image_data, width, height, samples_per_pixel);
}