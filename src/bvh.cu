//
// Created by Mario Ruiz on 5/18/21.
//

#include <cassert>
#include <algorithm>
#include "bvh.cuh"

#define SAH_CONSTRUCTION
#define NUM_BUCKETS 12


BVH::BVH(std::vector<Triangle> h_triangles)
{
    std::vector<BVHNode> h_nodes(2 * h_triangles.size());

    size_t index = 0;
    build(0, h_triangles.size(), index, h_triangles, h_nodes);

    cudaMalloc(&triangles, h_triangles.size() * sizeof(Triangle));
    cudaMemcpy(triangles, h_triangles.data(), h_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&nodes, h_nodes.size() * sizeof(BVHNode));
    cudaMemcpy(nodes, h_nodes.data(), h_nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
}

BVH::~BVH()
{
    cudaFree(triangles);
    cudaFree(nodes);
}

struct BucketInfo {
    AABB aabb;
    size_t count = 0;
};

static bool aabb_compare(const Triangle &a, const Triangle &b, size_t axis)
{
    float3 centroid_a = a.bounding_box().centroid();
    float3 centroid_b = b.bounding_box().centroid();
    return (&centroid_a.x)[axis] < (&centroid_b.x)[axis];   // Does this work???
}

static inline size_t bucket_idx(const AABB &box, const AABB &bound, int axis) {
    float3 centroid = box.centroid();
    return NUM_BUCKETS * ((&centroid.x)[axis] - (&bound.min.x)[axis]) / ((&bound.max.x)[axis] - (&bound.min.x)[axis]);
}

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
        std::nth_element(triangles.begin() + start, triangles.begin() + mid, triangles.begin() + end, compare);

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
