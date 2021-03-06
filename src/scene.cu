//
// Created by SuperMario210 on 6/7/2021.
//

#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <chrono>

#include "bvh.cuh"
#include "scene.cuh"
#include "render.cuh"

static void load_obj(const std::string &filename, std::vector<Triangle> &triangles)
{
    std::cout << "Loading .obj file...\n";
    std::ifstream obj_file (filename);
    std::vector<float3> vertices;
    vertices.emplace_back();
    size_t count = 0;
    std::string line;

    auto t1 = std::chrono::high_resolution_clock::now();
    while (!obj_file.eof()) {
        getline(obj_file, line);
        if (obj_file.eof()) { break; }

        char type;
        std::istringstream iss(line);
        iss >> type;

        if (type == 'v') {
            float3 v;
            iss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        } else if (type == 'f') {
            int i1, i2, i3;
            iss >> i1 >> i2 >> i3;
            i1 = (i1 < 0) ? i1 + vertices.size() : i1;
            i2 = (i2 < 0) ? i2 + vertices.size() : i2;
            i3 = (i3 < 0) ? i3 + vertices.size() : i3;
            triangles.emplace_back(vertices[i1], vertices[i2], vertices[i3]);
            count++;
        } else {
            std::cerr << "A parsing error occurred while reading in "
                      << "the .obj file" << filename << "\n";
            return;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Loaded " << count << " triangles in " << ms_int.count() << " ms\n";
}

static std::istringstream read_line(std::ifstream &file, std::string &type)
{
    type.clear();
    std::string line;
    getline(file, line);
    std::istringstream iss(line);
    iss >> type;
    return iss;
}

static float3 read_float3(std::ifstream &file, const std::string &check_type)
{
    std::string type;
    float x, y, z;
    std::istringstream iss = read_line(file, type);
    assert(type == check_type);
    iss >> x >> y >> z;
    return make_float3(x, y, z);
}

static float read_float(std::ifstream &file, const std::string &check_type)
{
    std::string type;
    float f;
    std::istringstream iss = read_line(file, type);
    assert(type == check_type);
    iss >> f;
    return f;
}

static std::string read_string(std::ifstream &file, const std::string &check_type)
{
    std::string type;
    std::istringstream iss = read_line(file, type);
    assert(type == check_type);
    iss >> type;
    return type;
}

static void read_material(std::ifstream &file, float4 &mat, uint &mat_type)
{
    std::string material_type = read_string(file, "material_type");
    if (material_type == "LAMBERTIAN")
    {
        float3 albedo = read_float3(file, "albedo");
        mat = make_float4(albedo, 0);
        mat_type = IS_DIFFUSE;
    }
    else if (material_type == "MIRROR")
    {
        float3 albedo = read_float3(file, "albedo");
        mat = make_float4(albedo, 0);
        mat_type = IS_MIRROR;
    }
    else if (material_type == "GLASS")
    {
        float3 albedo = read_float3(file, "albedo");
        float refraction_index = read_float(file, "refraction_index");
        mat = make_float4(albedo, refraction_index);
        mat_type = IS_GLASS;
    }
    else if (material_type == "GLOSSY")
    {
        float3 albedo = read_float3(file, "albedo");
        float3 specular = read_float3(file, "specular");
        float glossyness = read_float(file, "glossyness");
        mat = make_float4(albedo, glossyness);
        mat_type = IS_GLOSSY;
    }
//    else if (material_type == "PHONG")
//    {
//        float3 albedo = read_float3(file, "albedo");
//        float exponent = read_float(file, "exponent");
//        mat = make_float4(albedo, exponent);
//        mat_type = IS_PHONG;
//    }
    else {
        std::cerr << "Invalid material type: " << material_type << "\n";
    }
}

Scene::Scene(const std::string &filename, size_t width, size_t height) : camera(), environment(), spheres(), planes(),
                                                                         bvh(), image_data(), h_environment(), h_bvh()
{
    std::vector<Sphere> h_spheres;
    std::vector<Plane> h_planes;
    std::ifstream scene_file(filename);
    std::string type;

    while (!scene_file.eof()) {
        std::istringstream iss = read_line(scene_file, type);
        if (scene_file.eof()) { break; }

        if (type == "CAMERA") {
            // Only one camera allowed per scene
            assert(camera == nullptr);

            float3 position = read_float3(scene_file, "position");
            float3 look_at = read_float3(scene_file, "look_at");
            float fov = read_float(scene_file, "fov");
            float aperture = read_float(scene_file, "aperture");
            float focus_dist = read_float(scene_file, "focus_dist");

            Camera h_camera(position, look_at, fov, static_cast<float>(width) / height, aperture, focus_dist);
            gpuErrchk(cudaMalloc(&camera, sizeof(Camera)));
            gpuErrchk(cudaMemcpy(camera, &h_camera, sizeof(Camera), cudaMemcpyHostToDevice));
        } else if (type == "ENVIRONMENT") {
            // Only one environment allowed per scene
            assert(environment == nullptr);

            std::string file_name = read_string(scene_file, "file_name");

            h_environment = new EnvironmentMap(file_name);
            gpuErrchk(cudaMalloc(&environment, sizeof(EnvironmentMap)));
            gpuErrchk(cudaMemcpy(environment, h_environment, sizeof(EnvironmentMap), cudaMemcpyHostToDevice));
        } else if (type == "SPHERE") {
            float4 mat;
            uint mat_type;
            read_material(scene_file, mat, mat_type);
            float3 position = read_float3(scene_file, "position");
            float radius = read_float(scene_file, "radius");
            h_spheres.emplace_back(position, radius, mat, mat_type);
        } else if (type == "PLANE") {
            float4 mat;
            uint mat_type;
            read_material(scene_file, mat, mat_type);
            float3 position = read_float3(scene_file, "position");
            float3 normal = read_float3(scene_file, "normal");
            h_planes.emplace_back(position, normal, mat, mat_type);
        } else if (type == "OBJ") {
            // Only one obj allowed per scene (for now)
            assert(bvh == nullptr);

            float4 mat;
            uint mat_type;
            read_material(scene_file, mat, mat_type);
            std::string file_name = read_string(scene_file, "file_name");

            std::vector<Triangle> triangles;
            load_obj(file_name, triangles);

            h_bvh = new BVH(triangles, mat, mat_type);
            gpuErrchk(cudaMalloc(&bvh, sizeof(BVH)));
            gpuErrchk(cudaMemcpy(bvh, h_bvh, sizeof(BVH), cudaMemcpyHostToDevice));
        }
    }

    num_planes = h_planes.size();
    if (num_planes > 0) {
        gpuErrchk(cudaMalloc(&planes, num_planes * sizeof(Plane)));
        gpuErrchk(cudaMemcpy(planes, h_planes.data(), num_planes * sizeof(Plane), cudaMemcpyHostToDevice));
    }

    num_spheres = h_spheres.size();
    if (num_spheres > 0) {
        gpuErrchk(cudaMalloc(&spheres, num_spheres * sizeof(Sphere)));
        gpuErrchk(cudaMemcpy(spheres, h_spheres.data(), num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice));
    }

    gpuErrchk(cudaMalloc(&image_data, width * height * sizeof(float3)));

    assert(camera != nullptr);
    assert(environment != nullptr);
    assert(image_data != nullptr);
}

Scene::~Scene()
{
    gpuErrchk(cudaFree(camera));
    gpuErrchk(cudaFree(environment));
    gpuErrchk(cudaFree(image_data));
    if (planes != nullptr) gpuErrchk(cudaFree(planes));
    if (spheres != nullptr) gpuErrchk(cudaFree(spheres));
    if (bvh != nullptr) gpuErrchk(cudaFree(bvh));

    delete h_environment;
    if (h_bvh != nullptr) delete h_bvh;
}
