//
// Created by Mario Ruiz on 5/11/21.
//

#ifndef PATHTRACER_OBJECT_H
#define PATHTRACER_OBJECT_H


#include "common.h"
#include "material.cuh"
#include "aabb.h"

class Sphere
{
public:
    float3 position;
    float radius;

    __device__ Sphere(const float3 &position, float radius) : position(position), radius(radius)
    {

    }

    __device__ inline void intersect(const float3 &o, const float3 &d, float &t_max, float3 &n) const
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
            }
        }
    }
};

class Plane
{
public:
    float3 position, normal;

    __device__ Plane(const float3 &position, const float3 &normal) : position(position), normal(normalize(normal))
    {

    }

    __device__ __inline__ void intersect(const float3 &o, const float3 &d, float &t_max, float3 &n) const
    {
        float det = dot(d, normal);
        if (det < -EPSILON || det > EPSILON) {
            float t = dot(position - o, normal) / det;
            if (t > EPSILON && t < t_max) {
                t_max = t;
                n = normal;
            }
        }
    }
};


#endif //PATHTRACER_OBJECT_H
