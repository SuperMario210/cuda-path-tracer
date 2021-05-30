//
// Created by Mario Ruiz on 5/11/21.
//

#ifndef PATHTRACER_CAMERA_H
#define PATHTRACER_CAMERA_H


#include "ray.cuh"
#include "util.cuh"

class Camera
{
private:
    float3 origin;        // Camera position
    float3 bottom_left;   // Bottom left corner of the screen plane
    float3 right, up;     // Camera basis vectors
    float3 u, v, w;
    float lens_radius;

public:
    __host__ Camera(const float3 &origin, const float3 &look_at, float fov, float aspect_ratio, float aperture,
                    float focus_dist) : origin(origin)
    {
        // Calculate screen plane dimensions
        auto height = 2.0 * tan(fov * PI / 360);
        auto width = aspect_ratio * height;

        w = normalize(origin - look_at);
        u = normalize(cross(make_float3(0, 1, 0), w));
        v = cross(w, u);

        // Calculate camera basis vectors
        right = u * focus_dist * width;
        up = v * focus_dist * height;
        bottom_left = -right / 2 - up / 2 - w * focus_dist;
        lens_radius = aperture / 2;
    }

    __device__ inline Ray cast_ray(float s, float t) const
    {
        // Generate random vector in unit disc
//        float3 rd;
//        do {
//            rd = make_float3(randomf(-1,1), randomf(-1,1), 0);
//
//        } while (dot(rd, rd) >= 1);
//        rd *= lens_radius;
//
//        float3 offset = u * rd.x + v * rd.y;
//        return {origin + offset, bottom_left + right * s + up * t - offset};
        return {origin, bottom_left + right * s + up * t};
    }
};


#endif //PATHTRACER_CAMERA_H
