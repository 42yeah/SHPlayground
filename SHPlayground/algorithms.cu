#include "algorithms.cuh"
#include <glm/gtc/matrix_transform.hpp>
#include <random>
#include <glm/gtc/random.hpp>
#include "Triangle.cuh"

using namespace cuda;


namespace cuda {

    __device__ int doublefac(int i) {
        int result = 1;
        while (i > 1) {
            result *= i;
            i -= 2;
        }
        return result;
    }

    __device__ int fac(int i) {
        int result = 1;
        while (i > 1) {
            result *= i;
            i--;
        }
        return result;
    }

    __device__ float legendre(int l, int m, float x) {
        // rule 2
        float pmm = pow(-1, m) * doublefac(2 * m - 1) * sqrtf(pow(1 - x * x, (float) m));
        if (l == m) {
            return pmm;
        }
        // use rule 3 once
        float pmmp1 = x * (2.0f * m + 1) * pmm;
        if (l == m + 1) {
            return pmmp1;
        }
        for (int i = m + 2; i <= l; i++) {
            float pim = (x * (2 * i - 1) * pmmp1 - (i + m - 1) * pmm) / (i - m);
            pmm = pmmp1;
            pmmp1 = pim;
        }
        return pmmp1;
    }

    __device__ float sh_k(int l, int m) {
        return sqrtf((2 * l + 1) / (4.0f * glm::pi<float>()) * ((float) fac(l - abs(m)) / fac(l + abs(m))));
    }

    __device__ float sh(int l, int m, float theta, float phi) {
        float sqrt2 = sqrtf(2.0f);
        if (m > 0) {
            return sqrt2 * sh_k(l, m) * cosf(m * phi) * legendre(l, m, cosf(theta));
        }
        if (m < 0) {
            return sqrt2 * sh_k(l, m) * sinf(-m * phi) * legendre(l, -m, cosf(theta));
        }
        return sh_k(l, 0) * legendre(l, 0, cosf(theta));
    }

    __global__ void spherical_harmonics(glm::vec3* data, glm::ivec2 size, int l, int m) {
        int y = blockIdx.x;
        int x = threadIdx.x;

        glm::vec2 uv = glm::vec2((float) x, (float) y);
        
        uv.x /= (float) size.x;
        uv.y /= (float) size.y;

        // uniformly sample a sphere
        uv.y *= glm::pi<float>(); // theta
        uv.x *= 2.0f * glm::pi<float>(); // phi

        float shvalue = sh(l, m, uv.y, uv.x);

        data[y * size.x + x] = glm::vec3(shvalue, shvalue, shvalue);
    }

    __global__ void sh_project(glm::vec3* sh, glm::vec4* envmap, glm::ivec2 sh_size, glm::ivec2 envmap_size, glm::vec3 *result) {
        int y = threadIdx.x;
        
        glm::vec3 contrib(0.0f);
        for (int x = 0; x < sh_size.x; x++) {
            glm::vec2 uv = glm::vec2((float) x, (float) y);

            // 1. just put it to the center (we can't random)
            uv += glm::vec2(0.5f, 0.5f);

            // 2. normalize
            uv /= sh_size;

            // 3. project to spherical coordinate
            float theta = 2.0f * acosf(sqrtf(1.0f - uv.y));
            float phi = 2.0f * glm::pi<float>() * uv.x;
            
            // 4. normalize... again
            theta /= glm::pi<float>();
            phi = phi / 2.0f / glm::pi<float>();

            // 5. sample textures w.r.t. to the sh and envmap
            glm::ivec2 sample_sh((int) (phi * sh_size.x), (int) (theta * sh_size.y));
            // glm::ivec2 sample_sh((int) (uv.x * sh_size.x), (int) (uv.y * sh_size.y));
            glm::ivec2 sample_envmap((int) (phi * envmap_size.x), (int) (theta * envmap_size.y));
            // glm::ivec2 sample_envmap((int) (uv.x * envmap_size.x), (int) (uv.y * envmap_size.y));

            glm::vec3 shvalue = sh[sample_sh.y * sh_size.x + sample_sh.x];
            glm::vec3 envmap_value = glm::vec3(envmap[sample_envmap.y * envmap_size.x + sample_envmap.x]);

            // reinhart tonemapping
            envmap_value = envmap_value / (envmap_value + 1.0f);
            // gamma correction
            // envmap_value = glm::pow(envmap_value, glm::vec3(1.0f / 2.2f));

            contrib += shvalue * envmap_value;
        }
        // is this uniform? let's just assume it is
        contrib /= sh_size.x;

        result[y] = contrib;
    }

    __global__ void weight(glm::vec3 *a, glm::vec3 weight, glm::ivec2 size) {
        int x = threadIdx.x;
        int y = blockIdx.x;
        a[y * size.x + x] *= weight;
    }

    __global__ void add_to(glm::vec3 *a, glm::vec3 *b, glm::ivec2 size) {
        int x = threadIdx.x;
        int y = blockIdx.x;
        a[y * size.x + x] += b[y * size.x + x];
    }

    __global__ void vertex_sh_project_unshadowed(glm::vec3* sh, Vertex *vertices, glm::ivec2 sh_size, glm::vec3* result, int num_vertices, int block_offset) {
        int i = blockIdx.x;
        int y = threadIdx.x;

        int index = block_offset + i;
        if (index >= num_vertices) {
            return;
        }

        Vertex &vertex = vertices[index];
        int result_offset = index * sh_size.y;

        for (int x = 0; x < sh_size.x; x++) {
            glm::vec2 uv = glm::vec2((float) x, (float) y);

            // 1. just put it to the center (we can't random)
            uv += glm::vec2(0.5f, 0.5f);

            // 2. normalize
            uv /= sh_size;

            // 3. project to spherical coordinate
            float theta = 2.0f * acosf(sqrtf(1.0f - uv.y));
            float phi = 2.0f * glm::pi<float>() * uv.x;

            // 3.5 project to cartesian coordinate
            glm::vec3 cartesian(sinf(theta) * cosf(phi), cosf(theta), -sinf(theta) * sinf(phi));

            // 4. normalize... again
            theta /= glm::pi<float>();
            phi = phi / 2.0f / glm::pi<float>();

            // 5. now sample spherical coords
            glm::ivec2 sample_sh((int) (phi * sh_size.x), (int) (theta * sh_size.y));

            // 6. dot with the vertex normal & object color
            glm::vec3 lambertian = glm::clamp(glm::dot(vertex.normal, cartesian) * glm::vec3(1.0f), 0.0f, 1.0f);
            glm::vec3 shvalue = sh[sample_sh.y * sh_size.x + sample_sh.x];

            // 7. project lambertian onto SH basis
            result[result_offset + y] += lambertian * shvalue;
        }

        result[result_offset + y] /= sh_size.x;
    }


}

