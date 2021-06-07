#include "camera.cuh"
#include "render.cuh"

#define IMPORTANCE_SAMPLING
//#define RUSSIAN_ROULETTE
#define MIN_DEPTH               3
#define MAX_DEPTH               16

__global__ void intersect_scene(BVH *bvh, PathData *paths)
{
    const uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= MAX_PATHS || !paths->get_flag(index, IS_ACTIVE)) return;

    // Load ray data
    const float3 o = make_float3(paths->origin[index]);
    const float3 d = make_float3(paths->direction[index]);
    float t_max = FLT_MAX;
    float3 n;
    float4 mat;
    uint mat_type;

    // Intersect planes
//    const float3 position = make_float3(0, -0.283, 0);
    const float3 position = make_float3(0, 0, 0);
    const float3 normal = make_float3(0, 1, 0);
    const Plane plane(position, normal, make_float4(0.325, 0.3, 0.35, 0), IS_DIFFUSE);
    plane.intersect(o, d, t_max, n, mat, mat_type);

    // Intersect spheres
    const Sphere sphere1(make_float3(2.025, 0.5, 0), 0.5, make_float4(0.15, 0.25, 0.4, 0.05), IS_GLOSSY);
    const Sphere sphere2(make_float3(0.675, 0.5, 0), 0.5, make_float4(0.5, 0.5, 0.5, 0), IS_MIRROR);
    const Sphere sphere3(make_float3(-0.675, 0.5, 0), 0.5, make_float4(0.7, 0.7, 0.7, 0), IS_DIFFUSE);
    const Sphere sphere4(make_float3(-2.025, 0.5, 0), 0.5, make_float4(1, 1, 1, 1.5), IS_GLASS);
    sphere1.intersect(o, d, t_max, n, mat, mat_type);
    sphere2.intersect(o, d, t_max, n, mat, mat_type);
    sphere3.intersect(o, d, t_max, n, mat, mat_type);
    sphere4.intersect(o, d, t_max, n, mat, mat_type);

    // Intersect triangles
//    bvh->intersect(o, d, t_max, n, mat, mat_type);

    paths->direction[index].w = t_max;
    paths->normal[index] = make_float4(normalize(n));
    paths->material[index] = mat;
    paths->flags[index] = mat_type | IS_ACTIVE;
}

__device__ bool g_is_working = false;

__global__ void logic_kernel(PathData *paths, EnvironmentMap *envmap, float3 *image_data, uint samples_per_pixel, int seed)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= MAX_PATHS || !paths->get_flag(index, IS_ACTIVE)) return;

//    paths->flags[index] = 0;
    g_is_working = true;

    curandState rand_state;
    curand_init(seed + index, 0, 0, &rand_state);

    // Background intersection
    if (paths->direction[index].w == FLT_MAX || paths->depth[index]++ >= MAX_DEPTH) {
        uint pixel_index = paths->pixel_index[index];
        float3 color = make_float3(paths->throughput[index]) * envmap->sample(make_float3(paths->direction[index])) / samples_per_pixel;
        atomicAdd(&image_data[pixel_index].x, color.x);
        atomicAdd(&image_data[pixel_index].y, color.y);
        atomicAdd(&image_data[pixel_index].z, color.z);

        paths->flags[index] = IS_NEW_PATH;
        return;
    }

    // Material Intersection
//    if (paths->get_flag(index, IS_DIFFUSE)) {
//
//    }

#ifdef RUSSIAN_ROULETTE

    if (paths->depth[index]++ >= MIN_DEPTH) {
        float4 throughput = paths->throughput[index];
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        if (curand_uniform(&rand_state) > p) {
            paths->flags[index] = IS_NEW_PATH;
            return;
        }
        paths->throughput[index] *= 1 / p;
    }

#endif
}

__device__ uint g_path_count = 0;

__global__ void generate_primary_paths(Camera *camera, uint width, uint height, uint samples_per_pixel, uint path_count,
                                       PathData *paths, int seed, bool override)
{
    const uint index = blockDim.x * blockIdx.x + threadIdx.x;
//    if (index >= MAX_PATHS || !paths->get_flag(index, IS_NEW_PATH)) return;

    if (index >= MAX_PATHS || (!override && !paths->get_flag(index, IS_NEW_PATH))) return;

    const uint global_index = atomicAdd(&g_path_count, 1) / samples_per_pixel;
    if (global_index >= width * height) return;

    curandState rand_state;
    curand_init(seed + index, 0, 0, &rand_state);

    const uint x = (global_index) % width;
    const uint y = (global_index / width) % height;

    const float u = (float(x) + curand_uniform(&rand_state) - .5f) / float(width - 1);
    const float v = (float(y) + curand_uniform(&rand_state) - .5f) / float(height - 1);

    Ray r = camera->cast_ray(u, v, rand_state);
    paths->pixel_index[index] = (height - y - 1) * width + x;
    paths->origin[index] = make_float4(r.origin, EPSILON);
    paths->direction[index] = make_float4(r.direction, FLT_MAX);
    paths->throughput[index] = make_float4(1);
    paths->depth[index] = 0;
    paths->set_flag(index, IS_ACTIVE);
}

__global__ void generate_diffuse_paths(EnvironmentMap *envmap, PathData *paths, int seed)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= MAX_PATHS || !paths->get_flag(index, IS_DIFFUSE)) return;

    curandState rand_state;
    curand_init(seed + index, 0, 0, &rand_state);

    float3 norm = make_float3(paths->normal[index]);
    float3 out_dir = (curand_uniform(&rand_state) < 0.5) ? diffuse(norm, rand_state) : envmap->sample_lights(rand_state);

    float4 dir = paths->direction[index];
    dir *= dir.w;
    dir.w = 0;

    paths->origin[index] += dir;
    paths->direction[index] = make_float4(out_dir, FLT_MAX);

    float env_pdf = envmap->pdf(out_dir);
    float diff_pdf = max(dot(norm, out_dir) / PI, 0.0f);
    float mixed_pdf = (env_pdf + diff_pdf) * 0.5f;
    paths->throughput[index] *= paths->material[index] * (diff_pdf / mixed_pdf);
    paths->set_flag(index, IS_ACTIVE);
}

__global__ void generate_glossy_paths(EnvironmentMap *envmap, PathData *paths, int seed)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= MAX_PATHS || !paths->get_flag(index, IS_GLOSSY)) return;

    curandState rand_state;
    curand_init(seed + index, 0, 0, &rand_state);

    float4 dir = paths->direction[index];
    float3 norm = make_float3(paths->normal[index]);
    float4 offset = dir * dir.w;
    offset.w = 0;
    paths->origin[index] += offset;

    if (curand_uniform(&rand_state) < 0.05) {
        float3 out_dir = reflect(make_float3(dir), norm);
        paths->direction[index] = make_float4(out_dir, FLT_MAX);
        return;
    }

    float3 out_dir = (curand_uniform(&rand_state) < 0.5) ? diffuse(norm, rand_state) : envmap->sample_lights(rand_state);

    paths->direction[index] = make_float4(out_dir, FLT_MAX);

    float env_pdf = envmap->pdf(out_dir);
    float diff_pdf = max(dot(norm, out_dir) / PI, 0.0f);
    float mixed_pdf = (env_pdf + diff_pdf) * 0.5f;
    paths->throughput[index] *= paths->material[index] * (diff_pdf / mixed_pdf);
    paths->set_flag(index, IS_ACTIVE);
}

__global__ void generate_mirror_paths(PathData *paths)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= MAX_PATHS || !paths->get_flag(index, IS_MIRROR)) return;

    float4 dir = paths->direction[index];
    float3 norm = make_float3(paths->normal[index]);
    float3 out_dir = reflect(make_float3(dir), norm);

    dir *= dir.w;
    dir.w = 0;

    paths->origin[index] += dir;
    paths->direction[index] = make_float4(out_dir, FLT_MAX);
    paths->throughput[index] *= paths->material[index];
    paths->set_flag(index, IS_ACTIVE);
}

__global__ void generate_glass_paths(PathData *paths, int seed)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= MAX_PATHS || !paths->get_flag(index, IS_GLASS)) return;

    curandState rand_state;
    curand_init(seed + index, 0, 0, &rand_state);

    float3 dir = make_float3(paths->direction[index]);
    float3 norm = make_float3(paths->normal[index]);
    float3 offset = dir * paths->direction[index].w;
    paths->origin[index] += make_float4(offset);

    bool external = dot(dir, norm) < 0;
    norm = external ? norm : -norm;
    float ref_idx = external ? (1 / paths->material[index].w) : paths->material[index].w;
    float cos_t = fmin(dot(-dir, norm), 1.0f);
    float sin_t = sqrtf(1.0f - cos_t * cos_t);
    float3 out_dir;

    if (ref_idx * sin_t > 1.0f || reflectance(cos_t, ref_idx) > curand_uniform(&rand_state)) {
        out_dir = reflect(dir, norm);
    } else {
        out_dir = refract(dir, norm, ref_idx);
    }

    paths->direction[index] = make_float4(out_dir, FLT_MAX);
    paths->throughput[index] *= paths->material[index];
    paths->set_flag(index, IS_ACTIVE);
}

__host__ void launch_render_kernel(BVH *bvh, EnvironmentMap *envmap, Camera *camera, float3 *image_data, size_t width,
                                   size_t height, size_t samples_per_pixel, dim3 grid, dim3 block, PathData *paths)
{
    const uint block_size = 128;
    const uint grid_size = (MAX_PATHS + block_size - 1) / block_size;

    uint path_count = 0;
    bool is_working = false;
    bool override = true;
    do {
        generate_primary_paths<<<grid_size, block_size>>>(camera, width, height, samples_per_pixel,
                                                          path_count, paths, rand(), override);

        generate_diffuse_paths<<<grid_size, block_size>>>(envmap, paths, rand());
        generate_mirror_paths<<<grid_size, block_size>>>(paths);
        generate_glossy_paths<<<grid_size, block_size>>>(envmap, paths, rand());
        generate_glass_paths<<<grid_size, block_size>>>(paths, rand());

        intersect_scene<<<grid_size, block_size>>>(bvh, paths);

        logic_kernel<<<grid_size, block_size>>>(paths, envmap, image_data, samples_per_pixel, rand());
        override = false;

        bool temp = false;
        gpuErrchk(cudaMemcpyFromSymbol(&is_working, g_is_working, sizeof(bool)));
        gpuErrchk(cudaMemcpyToSymbol(g_is_working, &temp, sizeof(bool)));
    } while (is_working);
}