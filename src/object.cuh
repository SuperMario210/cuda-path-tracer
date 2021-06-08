//
// Created by Mario Ruiz on 5/11/21.
//

#ifndef PATHTRACER_OBJECT_H
#define PATHTRACER_OBJECT_H


#include "common.h"
#include "material.cuh"
#include "aabb.h"

class Sphere {
public:
    float3 position;
    float radius;
    float4 material;
    uint material_type;

    __device__ __inline__
    Sphere(const float3 &pos, float rad, const float4 &mat, uint mat_type) : position(pos), radius(rad), material(mat),
                                                                             material_type(mat_type)
    {

    }

    __device__ __inline__
    void intersect(const float3 &o, const float3 &d, float &t_max, float3 &n, float4 &mat, uint &mat_type) const
    {
        const float3 op = position - o;
        const float b = dot(op, d);
        float disc = b * b - dot(op, op) + radius * radius;

        if (disc > 0) {
            disc = sqrtf(disc);
            const float t = (b - disc) >= 0 ? (b - disc) : (b + disc);
            if (t > EPSILON && t < t_max) {
                t_max = t;
                n = (o + d * t - position) / radius;
                mat = material;
                mat_type = material_type;
            }
        }
    }
};

class Plane {
public:
    float3 position, normal;
    float4 material;
    uint material_type;

    __device__ __inline__
    Plane(const float3 &pos, const float3 &norm, const float4 &mat, const uint mat_type) : position(pos),
                                                                                           normal(normalize(norm)),
                                                                                           material(mat),
                                                                                           material_type(mat_type)
    {

    }

    __device__ __inline__
    void intersect(const float3 &o, const float3 &d, float &t_max, float3 &n, float4 &mat, uint &mat_type) const
    {
        float det = dot(d, normal);
        if (det < -EPSILON || det > EPSILON) {
            float t = dot(position - o, normal) / det;
            if (t > EPSILON && t < t_max) {
                t_max = t;
                n = normal;
                mat = material;
                mat_type = material_type;
            }
        }
    }
};


#endif //PATHTRACER_OBJECT_H
