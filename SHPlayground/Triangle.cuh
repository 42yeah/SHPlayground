#pragma once

#include "Vertex.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


struct Ray {
    // ray origin
    glm::vec3 origin;

    // ray direction
    glm::vec3 direction;

    // intersection range
    glm::vec2 intersect_range;
};


struct Intersection {
    __device__ __host__ Intersection(glm::vec3 position, float dist) : 
        intersected(true), position(position), dist(dist) {

    }

    __device__ __host__ Intersection() : intersected(false), position(glm::vec3(0.0f)), dist(0.0f) {

    }

    __device__ __host__ bool operator()() const {
        return intersected;
    }

    bool intersected;

    // intersection position
    glm::vec3 position;

    // distance from origin
    float dist; 
};


struct Triangle {
    __device__ __host__ Intersection intersect(const Ray &ray) const {
        // ray-plane intersection
        glm::vec3 e1 = b.position - a.position, e2 = c.position - a.position;
        glm::vec3 s = ray.origin - a.position;
        const glm::vec3 &d = ray.direction;

        glm::vec3 uvt = glm::vec3(
            -glm::dot(glm::cross(s, e2), d),
            glm::dot(glm::cross(e1, d), s),
            -glm::dot(glm::cross(s, e2), e1)
        ) / (glm::dot(glm::cross(e1, d), e2));
        
        float w = 1.0f - uvt.x - uvt.y;
        
        if (uvt.z < ray.intersect_range.x || uvt.z > ray.intersect_range.y ||
            uvt.x < 0.0f || uvt.x > 1.0f || uvt.y < 0.0f || uvt.y > 1.0f || w < 0.0f || w > 1.0f) {
            return Intersection();
        }

        return Intersection(ray.origin + uvt.z * ray.direction, uvt.z);
    }

    Vertex a, b, c;
};
