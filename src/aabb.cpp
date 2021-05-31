//
// Created by Mario Ruiz on 5/18/21.
//

#include <cfloat>
#include "aabb.h"

AABB::AABB() : min(make_float3(FLT_MAX)), max(make_float3(-FLT_MAX)) {}

AABB::AABB(const AABB &box0, const AABB &box1) : min(make_float3(
                                                     fminf(box0.min.x, box1.min.x),
                                                     fminf(box0.min.y, box1.min.y),
                                                     fminf(box0.min.z, box1.min.z))),
                                                 max(make_float3(
                                                     fmaxf(box0.max.x, box1.max.x),
                                                     fmaxf(box0.max.y, box1.max.y),
                                                     fmaxf(box0.max.z, box1.max.z)))
{
}

float AABB::surface_area() const
{
    float3 d = max - min;
    return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
}

float3 AABB::centroid() const
{
    return (min + max) / 2;
}

size_t AABB::max_extent() const
{
    float3 d = max - min;
    if (d.x >= d.y && d.x >= d.z) {
        return 0;
    } else if (d.y >= d.x && d.y >= d.z) {
        return 1;
    } else {
        return 2;
    }
}

void AABB::extend(const float3 &point)
{
    min.x = fminf(min.x, point.x);
    min.y = fminf(min.y, point.y);
    min.z = fminf(min.z, point.z);

    max.x = fmaxf(max.x, point.x);
    max.y = fmaxf(max.y, point.y);
    max.z = fmaxf(max.z, point.z);
}


void AABB::extend(const AABB &box)
{
    min.x = fminf(min.x, box.min.x);
    min.y = fminf(min.y, box.min.y);
    min.z = fminf(min.z, box.min.z);

    max.x = fmaxf(max.x, box.max.x);
    max.y = fmaxf(max.y, box.max.y);
    max.z = fmaxf(max.z, box.max.z);
}
