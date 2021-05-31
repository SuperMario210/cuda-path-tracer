//
// Created by Mario Ruiz on 5/18/21.
//

#ifndef PATHTRACER_AABB_H
#define PATHTRACER_AABB_H


#include "../include/cutil_math.cuh"
#include "ray.cuh"

class AABB
{
public:
    float3 min, max;

    AABB();
    AABB(const float3 &min, const float3 &max) : min(min), max(max) {}
    AABB(const AABB &box0, const AABB &box1);

    void extend(const float3 &point);
    void extend(const AABB &box);
    float surface_area() const;
    float3 centroid() const;
    size_t max_extent() const;
};


#endif //PATHTRACER_AABB_H
