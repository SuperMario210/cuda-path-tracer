//
// Created by Mario Ruiz on 5/11/21.
//

#ifndef PATHTRACER_OBJECT_H
#define PATHTRACER_OBJECT_H


#include <vector>
#include <cfloat>
#include <memory>
#include "ray.cuh"
#include "util.cuh"

struct Intersection
{
    bool external;
    float t;
    float3 position;
    float3 normal;

    __device__ Intersection() : t(FLT_MAX), position(), normal(), external()
    {

    }

    __device__ inline void set_normal(const Ray &r, const float3 &out_norm)
    {
        external = dot(r.direction, out_norm) < 0;
        normal = external ? out_norm : -out_norm;
    }
};

class Sphere
{
public:
    float3 position;
    float radius;

    __device__ Sphere(const float3 &position, float radius) : position(position), radius(radius)
    {

    }

    __device__ inline bool intersect(const Ray &r, Intersection &intersect) const
    {
        float3 diff = r.origin - position;
        auto r_len = dot(r.direction, r.direction);
        auto b = dot(diff, r.direction);
        auto c = dot(diff, diff) - radius * radius;
        auto det = b * b - r_len * c;

        if (det < 0) {
            return false;
        }

        auto sqrtd = sqrt(det);
        auto t = (-b - sqrtd) >= 0 ? (-b - sqrtd) / r_len : (-b + sqrtd) / r_len;
        if (t < EPSILON || t >= intersect.t) {
            return false;
        }

        intersect.t = t;
        intersect.position = r.at(intersect.t);
        intersect.set_normal(r, (intersect.position - position) / radius);
        return true;
    }
};

class Plane
{
public:
    float3 position, normal;

    __device__ Plane(const float3 &position, const float3 &normal) : position(position), normal(normalize(normal))
    {

    }

    __device__ inline bool intersect(const Ray &r, Intersection &intersect) const
    {
        auto d = dot(r.direction, normal);
        if (d < EPSILON && d > -EPSILON) {
            return false;
        }

        auto t = dot(position - r.origin, normal) / d;
        if (t < EPSILON || t >= intersect.t) {
            return false;
        }

        intersect.t = t;
        intersect.position = r.at(intersect.t);
        intersect.set_normal(r, normal);
        return true;
    }
};


#endif //PATHTRACER_OBJECT_H
