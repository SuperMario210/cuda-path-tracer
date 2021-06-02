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
    Material *material;

    __device__ Sphere(const float3 &position, float radius, Material *mat) : position(position), radius(radius), material(mat)
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
        intersect.material = material;
        return true;
    }
};

class Plane
{
public:
    float3 position, normal;
    Material *material;

    __device__ Plane(const float3 &position, const float3 &normal, Material *mat) : position(position), normal(normalize(normal)), material(mat)
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
        intersect.material = material;
        return true;
    }
};

class Triangle
{
public:
    float3 v0, v1, v2;
//    Material *material; // TODO: avoid storing per triangle

    __host__ __device__ Triangle(const float3 &v0, const float3 &v1, const float3 &v2) : v0(v0), v1(v1), v2(v2)
    {

    }

    __device__ inline bool intersect(const Ray &r, Intersection &intersect) const
    {
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;

        float3 pvec = cross(r.direction, e2);
        float det = dot(e1, pvec);

        if (det == 0) {
            return false;
        }
        float inv_det = 1.0f / det;

        float3 tvec = r.origin - v0;

        float u = dot(tvec, pvec) * inv_det;
        if (u < 0 || u > 1.0f) {
            return false;
        }

        float3 qvec = cross(tvec, e1);

        float v = dot(r.direction, qvec) * inv_det;
        if (v < 0 || u + v > 1.0f) {
            return false;
        }

        float t = dot(e2, qvec) * inv_det;
        if (t < EPSILON || t >= intersect.t) {
            return false;
        }

        intersect.t = t;
        intersect.position = r.at(intersect.t);
        intersect.set_normal(r, normalize(cross(e1, e2)));
//        intersect.material = material;
        return true;
    }

    AABB bounding_box() const
    {
        float3 min = fminf(fminf(v0, v1), v2);
        float3 max = fmaxf(fmaxf(v0, v1), v2);

        if (max.x - min.x == 0) max.x += EPSILON;
        if (max.y - min.y == 0) max.y += EPSILON;
        if (max.z - min.z == 0) max.z += EPSILON;

        return {min, max};
    }
};


#endif //PATHTRACER_OBJECT_H
