//
// Created by SuperMario210 on 5/30/2021.
//

#ifndef CUDA_BASE_MATERIAL_CUH
#define CUDA_BASE_MATERIAL_CUH


#include "util.cuh"
#include "intersection.cuh"

enum MaterialType
{
    LAMBERTIAN,
    MIRROR,
    GLASS,
    GLOSSY
};

class Material
{
private:
    __device__ static inline float3 diffuse(const float3 &n, curandState &rand_state) {
        // randomly generate point in sphere
        float z = curand_uniform(&rand_state) * 2.0f - 1.0f;
        float a = curand_uniform(&rand_state) * 2.0f * PI;
        float r = sqrtf(1.0f - z * z);
        float x = r * cosf(a);
        float y = r * sinf(a);
        float3 dir = make_float3(x, y, z) + n;

        if (dir.x < EPSILON && dir.y < EPSILON && dir.z < EPSILON)
            return n;

        return normalize(dir);
    }

    __device__ static inline float3 refract(const float3 &v, const float3 &n, float ref_idx) {
        // Refracts a given ray
        float cos_t = fmin(dot(-v, n), 1.0f);
        float3 perp = (v + n * cos_t) * ref_idx;
        float3 parallel = n * -sqrt(fabs(1.0f - dot(perp, perp)));
        return normalize(perp + parallel);
    }

    __device__ static inline float reflectance(float cos_t, float ref_idx) {
        // Schlick approximation for reflectance
        float r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * powf(1 - cos_t, 5);
    }

public:
    const MaterialType type;
    const float3 albedo;
    const float value;

    __host__ __device__ inline Material(MaterialType type, const float3 &albedo, float value) : type(type),
                                                                                                albedo(albedo),
                                                                                                value(value)
    {

    }

    __device__ inline Ray brdf(const Ray &in_ray, const Intersection &intersect, float3 &attenuation,
                               curandState &rand_state, bool &importance_sample) const
    {
        importance_sample = false;
        attenuation = albedo;

        if (type == LAMBERTIAN)
        {
            importance_sample = true;
            float3 dir = diffuse(intersect.normal, rand_state);
            return {intersect.position, dir};
        }
        else if (type == MIRROR)
        {
            return {intersect.position, reflect(in_ray.direction, intersect.normal)};
        }
        else if (type == GLASS)
        {
            float ref_idx = intersect.external ? (1 / value) : value;
            float cos_t = fmin(dot(-in_ray.direction, intersect.normal), 1.0f);
            float sin_t = sqrtf(1.0f - cos_t * cos_t);

            if (ref_idx * sin_t > 1.0f || reflectance(cos_t, ref_idx) > curand_uniform(&rand_state)) {
                return {intersect.position, reflect(in_ray.direction, intersect.normal)};
            } else {
                return {intersect.position, refract(in_ray.direction, intersect.normal, ref_idx)};
            }
        }
        else // type == GLOSSY
        {
            if (curand_uniform(&rand_state) < value) {
                attenuation = make_float3(1);
                return {intersect.position, reflect(in_ray.direction, intersect.normal)};
            } else {
                importance_sample = true;
                return {intersect.position, diffuse(intersect.normal, rand_state)};
            }
        }
    }

    __device__ inline float pdf(const Ray &in_ray, const Intersection &intersect, const Ray &sample) const
    {
        // This method should only ever be called if type == LAMBERTIAN or type == GLOSSY
        return max(dot(intersect.normal, sample.direction) / PI, 0.0f);
    }
};


#endif //CUDA_BASE_MATERIAL_CUH
