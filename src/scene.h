//
// Created by SuperMario210 on 6/7/2021.
//

#ifndef PATHTRACER_SCENE_H
#define PATHTRACER_SCENE_H


#include "image.h"
#include "environment_map.cuh"
#include "camera.cuh"

class Scene
{
public:
    // Stored on device
    Camera *camera;
    EnvironmentMap *environment;
    Sphere *spheres;
    uint num_spheres;
    Plane *planes;
    uint num_planes;
    BVH *bvh;
    float3 *image_data;

    // Stored on host
    EnvironmentMap *h_environment;
    BVH *h_bvh;

    Scene(const std::string &filename, size_t width, size_t height);
    ~Scene();
};


#endif //PATHTRACER_SCENE_H
