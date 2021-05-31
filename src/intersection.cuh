//
// Created by SuperMario210 on 5/30/2021.
//

#ifndef CUDA_BASE_INTERSECTION_CUH
#define CUDA_BASE_INTERSECTION_CUH

class Material;

struct Intersection
{
    bool external;
    float t;
    float3 position;
    float3 normal;
    Material *material;

    __device__ Intersection() : t(FLT_MAX), position(), normal(), external()
    {

    }

    __device__ inline void set_normal(const Ray &r, const float3 &out_norm)
    {
        external = dot(r.direction, out_norm) < 0;
        normal = external ? out_norm : -out_norm;
    }
};


#endif //CUDA_BASE_INTERSECTION_CUH
