//
// Created by Mario Ruiz on 5/18/21.
//

#include <cassert>
#include <algorithm>
#include <set>
#include <chrono>
#include "bvh.cuh"
#include "../include/linear_math.cuh"

#define SAH_CONSTRUCTION
#define NUM_BUCKETS 12

/**
 * Used for storing information about each bucket during SAH construction
 */
struct BucketInfo {
    AABB aabb;              // The bounding box for this bucket
    size_t count = 0;       // The number of triangles in this bucket
};

/**
 * Used for flattening the BVH before sending it to the GPU
 */
struct StackEntry {
    size_t index;           // The index of this entry in the nodes vector
    size_t parent_index;    // The index of this entry's parent node
    bool is_left;           // Is this entry a left or right child node
};

/**
 * A compressed version of the BVHNode for use on the GPU.  Stores the left and right bounding boxes and the indices of
 * the left and right child nodes in 64 bytes
 */
struct GPUBVHNode {
    float4 left_xy;         // (left.min.x, left.max.x, left.min.y, left.max.y)
    float4 right_xy;        // (right.min.x, right.max.x, right.min.y, right.max.y)
    float4 left_right_z;    // (left.min.z, left.max.z, right.min.z, right.max.z)
    int4 child_indices;     // (left.index, right.index, 0, 0)
};

BVH::BVH(std::vector<Triangle> h_triangles, const float4 &mat, const uint mat_type) : material(mat), material_type(mat_type)
{
    std::cout << "Constructing BVH...\n";
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<BVHNode> h_nodes(2 * h_triangles.size());
    size_t index = 0;
    build(0, h_triangles.size(), index, h_triangles, h_nodes);
    send_to_device(h_triangles, h_nodes);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Constructed BVH in " << ms_int.count() << " ms\n";
}

BVH::~BVH()
{
    cudaFree(triangles);
    cudaFree(nodes);

    cudaDestroyTextureObject(nodes_texture);
}

/**
 * Compares the centroids of two triangle's bounding boxes along the given axis
 * @param a the first triangle
 * @param b the second triangle
 * @param axis the axis along which to compare (0 = x, 1 = y, 2 = z)
 * @return true if a's centroid < b's centroid on the given axis, false otherwise
 */
static bool aabb_compare(const Triangle &a, const Triangle &b, size_t axis)
{
    float3 centroid_a = a.bounding_box().centroid();
    float3 centroid_b = b.bounding_box().centroid();
    return (&centroid_a.x)[axis] < (&centroid_b.x)[axis];
}

/**
 * Calculates what bucket a given bounding box should be placed into
 * @param box the bounding box to sort
 * @param bound the bounding box surrounding all objects being considered
 * @param axis the axis we are splitting along (0 = x, 1 = y, 2 = z)
 * @return the index of the bucket in which box should be placed
 */
static inline size_t bucket_idx(const AABB &box, const AABB &bound, int axis) {
    float3 centroid = box.centroid();
    return NUM_BUCKETS * ((&centroid.x)[axis] - (&bound.min.x)[axis]) / ((&bound.max.x)[axis] - (&bound.min.x)[axis]);
}

/**
 * Recursively builds a BVH from the given vector of triangles
 * @param start index in the triangles vector at which to start
 * @param end index in the triangles vector at which to end
 * @param index the current index in the BVHNodes vector
 * @param h_triangles a vector of triangles to convert into a BVH
 * @param h_nodes a vector of BVHNodes representing the completed BVH
 */
void BVH::build(size_t start, size_t end, size_t &index, std::vector<Triangle> &h_triangles, std::vector<BVHNode> &h_nodes)
{
    BVHNode *node = &h_nodes[index++];

    AABB centroid_bounds, bounds;
    for (size_t i = start; i < end; i++) {
        AABB box = h_triangles[i].bounding_box();
        bounds.extend(box);
        centroid_bounds.extend(box.centroid());
    }

    auto axis = centroid_bounds.max_extent();
    node->axis = axis;
    auto compare = [axis] (const Triangle &a, const Triangle &b) -> bool {
        return aabb_compare(a, b, axis);
    };

    size_t num_objects = end - start;
    if (num_objects == 1) {
        // Leaf node with one object
        node->num_children = 1;
        node->index = start;
        node->aabb = h_triangles[start].bounding_box();
    } else if (num_objects == 2) {
        // Leaf node with two triangles
        node->num_children = 2;
        // Ensure triangles are ordered according to axis
        if (!compare(h_triangles[start], h_triangles[start + 1])) {
            std::swap(h_triangles[start], h_triangles[start + 1]);
        }
        node->index = start;
        node->aabb = AABB(h_triangles[start].bounding_box(), h_triangles[start + 1].bounding_box());
    } else {
        // Interior node
        node->num_children = 0;

#ifdef SAH_CONSTRUCTION

        // Split triangles into buckets
        BucketInfo buckets[NUM_BUCKETS];
        for (size_t i = start; i < end; i++) {
            AABB box = h_triangles[i].bounding_box();
            int b = bucket_idx(box, centroid_bounds, axis);
            if (b == NUM_BUCKETS) b--;
            assert(b >= 0 && b < NUM_BUCKETS);
            buckets[b].count++;
            buckets[b].aabb.extend(box);
        }

        // Calculate SAH of each bucket split
        float costs[NUM_BUCKETS - 1];
        for (auto i = 0; i < NUM_BUCKETS - 1; i++) {
            AABB box0, box1;
            size_t num0 = 0, num1 = 0;
            for (auto j = 0; j <= i; j++) {
                box0.extend(buckets[j].aabb);
                num0 += buckets[j].count;
            }
            for (auto j = i + 1; j < NUM_BUCKETS; j++) {
                box1.extend(buckets[j].aabb);
                num1 += buckets[j].count;
            }
            costs[i] = (num0 * box0.surface_area() + num1 * box1.surface_area()) / bounds.surface_area() + 1;
        }

        // Get best bucket split
        float min_cost = costs[0];
        size_t min_idx = 0;
        for (auto i = 1; i < NUM_BUCKETS - 1; i++) {
            if (costs[i] < min_cost) {
                min_cost = costs[i];
                min_idx = i;
            }
        }

        // Split triangles according to best split
        auto mid_ptr = std::partition(h_triangles.begin() + start, h_triangles.begin() + end,
          [=] (const Triangle &tri) {
                AABB box = tri.bounding_box();
                int b = bucket_idx(box, centroid_bounds, axis);
                if (b == NUM_BUCKETS) b--;
                assert(b >= 0 && b < NUM_BUCKETS);
                return b <= min_idx;
            }
        );
        size_t mid = mid_ptr - h_triangles.begin();

#else

        // Split triangles into two equal groups
        size_t mid = start + num_objects / 2;
        std::nth_element(h_triangles.begin() + start, h_triangles.begin() + mid, h_triangles.begin() + end, compare);

#endif

        assert(mid < end && mid > start);

        // Build child nodes
        size_t left_idx = index;
        build(start, mid, index, h_triangles, h_nodes);
        node->index = index;
        build(mid, end, index, h_triangles, h_nodes);

        // Get bounding box
        node->aabb = AABB(h_nodes[left_idx].aabb, h_nodes[node->index].aabb);
    }

}

/**
 * Converts triangles into normalized form used in Woop's triangle intersection algorithm
 * @param triangle The triangle to "Woopify"
 * @param gpu_triangle_data The vector of triangle data to add the woopified triangle to
 */
void woopify_triangle(const Triangle &triangle, std::vector<float4> &gpu_triangle_data)
{
    // compute edges and transform them with a matrix
    Mat4f mtx;
    mtx.setCol(0, make_float4(triangle.v0 - triangle.v2, 0.0f));
    mtx.setCol(1, make_float4(triangle.v1 - triangle.v2, 0.0f));
    mtx.setCol(2, make_float4(cross(triangle.v0 - triangle.v2, triangle.v1 - triangle.v2), 0.0f));
    mtx.setCol(3, make_float4(triangle.v2, 1.0f));
    mtx = invert(mtx);

    gpu_triangle_data.push_back(make_float4(mtx(2, 0), mtx(2, 1), mtx(2, 2), -mtx(2, 3)));
    gpu_triangle_data.push_back(mtx.getRow(0));
    gpu_triangle_data.push_back(mtx.getRow(1));
}

void BVH::send_to_device(const std::vector<Triangle> &h_triangles, const std::vector<BVHNode> &h_nodes)
{
    std::vector<GPUBVHNode> gpu_nodes;
    gpu_nodes.reserve(h_nodes.size() / 2);
    std::vector<float4> gpu_triangle_data;
    gpu_triangle_data.reserve(h_triangles.size() * 4);
    gpu_triangle_data.push_back(make_float4(0));
    std::vector<StackEntry> stack;
    stack.push_back({0, 0, false});

    while (!stack.empty()) {
        StackEntry entry = stack[stack.size() - 1];
        stack.pop_back();

        const BVHNode *node = &h_nodes[entry.index];
        const BVHNode* left = &h_nodes[entry.index + 1];
        const BVHNode* right = &h_nodes[node->index];
        GPUBVHNode gpu_node;

        gpu_node.left_xy = make_float4(left->aabb.min.x, left->aabb.max.x, left->aabb.min.y, left->aabb.max.y);
        gpu_node.right_xy = make_float4(right->aabb.min.x, right->aabb.max.x, right->aabb.min.y, right->aabb.max.y);
        gpu_node.left_right_z = make_float4(left->aabb.min.z, left->aabb.max.z, right->aabb.min.z, right->aabb.max.z);
        gpu_node.child_indices = make_int4(0);

        if (entry.index != 0) {
            GPUBVHNode *parent_node = &gpu_nodes[entry.parent_index];
            if (entry.is_left) {
                parent_node->child_indices.x = gpu_nodes.size();
            } else {
                parent_node->child_indices.y = gpu_nodes.size();
            }
        }

        if (left->num_children == 0) {
            stack.push_back({entry.index + 1, gpu_nodes.size(), true});
        } else {
            gpu_node.child_indices.x = ~gpu_triangle_data.size();
            for (size_t i = 0; i < left->num_children; i++) {
                woopify_triangle(h_triangles[left->index + i], gpu_triangle_data);
            }

            gpu_triangle_data.push_back(make_float4(0));
            int4 end_marker = make_int4(0x80000000);
            memcpy(gpu_triangle_data.data() + gpu_triangle_data.size() - 1, &end_marker, sizeof(int4));
        }

        if (right->num_children == 0) {
            stack.push_back({node->index, gpu_nodes.size(), false});
        } else {
            gpu_node.child_indices.y = ~gpu_triangle_data.size();
            for (size_t i = 0; i < right->num_children; i++) {
                woopify_triangle(h_triangles[right->index + i], gpu_triangle_data);
            }

            gpu_triangle_data.push_back(make_float4(0));
            int4 end_marker = make_int4(0x80000000);
            memcpy(gpu_triangle_data.data() + gpu_triangle_data.size() - 1, &end_marker, sizeof(int4));
        }

        gpu_nodes.push_back(gpu_node);
    }

    gpuErrchk(cudaMalloc(&triangles, gpu_triangle_data.size() * sizeof(float4)));
    gpuErrchk(cudaMemcpy(triangles, gpu_triangle_data.data(), gpu_triangle_data.size() * sizeof(float4), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&nodes, gpu_nodes.size() * sizeof(GPUBVHNode)));
    gpuErrchk(cudaMemcpy(nodes, gpu_nodes.data(), gpu_nodes.size() * sizeof(GPUBVHNode), cudaMemcpyHostToDevice));

    // Specify texture
    cudaResourceDesc resource_desc{};
    memset(&resource_desc, 0, sizeof(resource_desc));
    resource_desc.resType = cudaResourceTypeLinear;
    resource_desc.res.linear.devPtr = triangles;
    resource_desc.res.linear.sizeInBytes = gpu_triangle_data.size() * sizeof(float4);
    resource_desc.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    // Specify texture object parameters
    cudaTextureDesc texture_desc{};
    memset(&texture_desc, 0, sizeof(texture_desc));

    // Create nodes texture object
    resource_desc.res.linear.devPtr = nodes;
    resource_desc.res.linear.sizeInBytes = gpu_nodes.size() * sizeof(GPUBVHNode);
    resource_desc.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    nodes_texture = 0;
    gpuErrchk(cudaCreateTextureObject(&nodes_texture, &resource_desc, &texture_desc, nullptr));
}
