#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glad/glad.h>
#include <vector>
#include "EnvMap.h"


class SHSampler {
public:
    SHSampler() : _num_bands(0), _size(0, 0), buffer(nullptr), _textures(nullptr), _reconstructed(0) {  }

    SHSampler(int num_bands);

    SHSampler(const SHSampler &) = delete;

    ~SHSampler() {
        if (buffer) {
            for (int i = 0; i < _num_bands * _num_bands; i++) {
                delete[] buffer[i];
            }
            delete[] buffer;
        }
        if (_textures) {
            glDeleteTextures(_num_bands * _num_bands, _textures);
            delete[] _textures;
        }
        if (_reconstructed != 0) {
            glDeleteTextures(1, &_reconstructed);
        }
    }

    // does not retain old buffer
    void resize(glm::ivec2 new_size, int num_bands = -1);

    glm::vec3 &operator()(int l, int m, float theta, float phi);

    inline int num_bands() {
        return _num_bands;
    }

    inline glm::ivec2 size() {
        return _size;
    }

    // fill all the RGB channels with corresponding spherical harmonic value
    void sample_sh();

    void visualize();

    inline const GLuint* textures() const {
        return _textures;
    }

    inline const std::vector<glm::vec3> &coefficients() const {
        return _coefficients;
    }

    void calc_coefficients(EnvMap &envmap);

    void reconstruct();

    GLuint reconstructed() {
        return _reconstructed;
    }

private:
    GLuint *_textures;
    glm::ivec2 _size;
    glm::vec3 **buffer;
    std::vector<glm::vec3> _coefficients;
    int _num_bands;

    GLuint _reconstructed;
};

