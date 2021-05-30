//
// Created by Mario Ruiz on 5/11/21.
//

#ifndef PATHTRACER_RAY_H
#define PATHTRACER_RAY_H


#include "../include/cutil_math.cuh"

class Ray
{
public:
    float3 origin, direction;

    __device__ inline Ray(const float3 &orig, const float3 &dir) : origin(orig), direction(dir)
    {

    }

    __device__ inline float3 at(float t) const
    {
        return origin + direction * t;
    }
};


#endif //PATHTRACER_RAY_H
