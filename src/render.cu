#include <cuda_runtime.h>
#include <vector_types.h>
#include "camera.cuh"
#include "object.cuh"
#include "environment_map.cuh"
#include "material.cuh"
#include <curand_kernel.h>

#define IMPORTANCE_SAMPLING
#define MAX_DEPTH   16


__device__ float3 path_trace(Ray r, curandState &rand_state, EnvironmentMap *envmap)
{
    Material plane_mat(LAMBERTIAN, make_float3(0.325, 0.3, 0.35), 0);
    Plane plane(make_float3(0), make_float3(0, 1, 0), &plane_mat);

    Material sphere1_mat(GLASS, make_float3(1), 1.5);
    Sphere sphere1(make_float3(-2.025, 0.5, 0), 0.5, &sphere1_mat);

    Material sphere2_mat(LAMBERTIAN, make_float3(0.7), 0);
    Sphere sphere2(make_float3(-0.675, 0.5, 0), 0.5, &sphere2_mat);

    Material sphere3_mat(MIRROR, make_float3(0.5), 0);
    Sphere sphere3(make_float3(0.675, 0.5, 0), 0.5, &sphere3_mat);

    Material sphere4_mat(GLOSSY, make_float3(0.15, 0.25, 0.4), 0.05);
    Sphere sphere4(make_float3(2.025, 0.5, 0), 0.5, &sphere4_mat);

    float3 color = make_float3(1);
    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        // Intersect scene
        Intersection intersect;
        bool hit0 = plane.intersect(r, intersect);
        bool hit1 = sphere1.intersect(r, intersect);
        bool hit2 = sphere2.intersect(r, intersect);
        bool hit3 = sphere3.intersect(r, intersect);
        bool hit4 = sphere4.intersect(r, intersect);

        if (hit0 || hit1 || hit2 || hit3 || hit4) {

#ifdef IMPORTANCE_SAMPLING

            float3 attenuation;
            bool importance_sample = false;
            Ray r_new = intersect.material->brdf(r, intersect, attenuation, rand_state, importance_sample);

            if (importance_sample) {
                if (curand_uniform(&rand_state) < 0.5) {
                    r_new.direction = envmap->sample_lights(rand_state);
                }

                float env_pdf = envmap->pdf(r_new.direction);
                float diff_pdf = intersect.material->pdf(r, intersect, r_new);
                float mixed_pdf = (env_pdf + diff_pdf) * 0.5f;
                color *= diff_pdf / mixed_pdf;
            }

            color *= attenuation;
            r = r_new;

#else

            float3 attenuation;
            bool importance_sample = false;
            r = intersect.material->brdf(r, intersect, attenuation, rand_state, importance_sample);
            color *= attenuation;

#endif

        } else {
            return color * envmap->sample(r.direction);
        }
    }

    return make_float3(0);
}

__global__ void render_kernel(EnvironmentMap *envmap, Camera *camera, float3 *image_data, size_t width, size_t height,
                              size_t samples_per_pixel)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = (height - y - 1) * width + x;

    curandState rand_state;
    curand_init(1984 + i, 0, 0, &rand_state);

    float3 accum_color = make_float3(0);
    for (int samp = 0; samp < samples_per_pixel; samp++) {
        float u = (float(x) + curand_uniform(&rand_state) - .5f) / float(width - 1);
        float v = (float(y) + curand_uniform(&rand_state) - .5f) / float(height - 1);

        Ray ray = camera->cast_ray(u, v, rand_state);
        accum_color += path_trace(ray, rand_state, envmap);
    }

    image_data[i] = accum_color / samples_per_pixel;
}