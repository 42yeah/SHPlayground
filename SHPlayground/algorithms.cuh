#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>


namespace cuda {
    __global__ void spherical_harmonics(glm::vec3 *data, glm::ivec2 size, int l, int m);

    __global__ void sh_project(glm::vec3 *sh, glm::vec4 *envmap, glm::ivec2 sh_size, glm::ivec2 envmap_size, glm::vec3 *result);

    __global__ void weight(glm::vec3 *a, glm::vec3 weight, glm::ivec2 size);

    // add b to a
    __global__ void add_to(glm::vec3 *a, glm::vec3 *b, glm::ivec2 size);
};
