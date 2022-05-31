#include "SHSampler.cuh"
#include "CudaPtr.cuh"
#include "algorithms.cuh"


SHSampler::SHSampler(int num_bands) : _num_bands(num_bands), _size(0, 0), buffer(nullptr), _textures(nullptr), _reconstructed(0) {

}

void SHSampler::resize(glm::ivec2 new_size, int num_bands) {
    if (buffer) {
        for (int i = 0; i < _num_bands * _num_bands; i++) {
            delete[] buffer[i];
        }
        delete[] buffer;
    }
    if (num_bands < 0) {
        num_bands = _num_bands;
    }
    buffer = new glm::vec3*[num_bands * num_bands];
    for (int i = 0; i < num_bands * num_bands; i++) {
        buffer[i] = new glm::vec3[new_size.x * new_size.y];
    }
    _size = new_size;
    _num_bands = num_bands;
    if (_textures) {
        glDeleteTextures(_num_bands * _num_bands, _textures);
        delete[] _textures;
    }
    _textures = new GLuint[_num_bands * _num_bands];
    glGenTextures(_num_bands * _num_bands, _textures);
    _coefficients.resize(_num_bands* _num_bands);
    for (int i = 0; i < _num_bands * _num_bands; i++) {
        _coefficients[i] = glm::vec3(0.0f);
    }
}

glm::vec3 &SHSampler::operator()(int l, int m, float theta, float phi) {
    float pi = glm::pi<float>();
    float pi_2 = 2.0f * pi;
    if (l < 0 || l >= _num_bands || m < -l || m > l || theta < 0.0f || theta > pi || phi < 0.0f || phi > pi_2) {
        throw "ERR! Trying to access either out of band, or out of range value";
    }
    int index = l * (l + 1) + m;

    // normalize
    theta = glm::clamp(theta / pi, 0.0f, 1.0f);
    phi = glm::clamp(phi / pi_2, 0.0f, 1.0f);

    int x = int(phi * _size.x);
    int y = int(theta * _size.y);

    return buffer[index][y * _size.x + x];
}

void SHSampler::sample_sh() {
    for (int l = 0; l < _num_bands; l++) {
        for (int m = -l; m <= l; m++) {
            int index = l * (l + 1) + m;
            
            {
                CudaPtr<glm::vec3> ptr(buffer[index], _size.x * _size.y);
                cuda::spherical_harmonics<<<_size.y, _size.x>>>(ptr(), _size, l, m);
            }
        }
    }
}

void SHSampler::visualize() {
    if (!_textures) {
        glGenTextures(_num_bands * _num_bands, _textures);
    }
    for (int l = 0; l < _num_bands; l++) {
        for (int m = -l; m <= l; m++) {
            int index = l * (l + 1) + m;
            glBindTexture(GL_TEXTURE_2D, _textures[index]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _size.x, _size.y, 0, GL_RGB, GL_FLOAT, buffer[index]);
        }
    }
}

void SHSampler::calc_coefficients(EnvMap &envmap) {
    CudaPtr<glm::vec4> envmap_cuda(envmap.data, envmap.size().x * envmap.size().y);
    std::vector<glm::vec3> result;
    result.resize(_size.y);

    for (int i = 0; i < _coefficients.size(); i++) {
        _coefficients[i] = glm::vec3(0.0f);
    }

    for (int l = 0; l < _num_bands; l++) {
        for (int m = -l; m <= l; m++) {
            int index = l * (l + 1) + m;
            for (int y = 0; y < _size.y; y++) {
                result[y] = glm::vec3(0.0f);
            }

            {
                CudaPtr<glm::vec3> sh_cuda(buffer[index], _size.x * _size.y);
                CudaPtr<glm::vec3> result_cuda(&result[0], result.size());

                cuda::sh_project<<<1, _size.y>>>(sh_cuda(), envmap_cuda(), _size, envmap.size(), result_cuda());
            }

            for (int i = 0; i < result.size(); i++) {
                _coefficients[index] += result[i];
            }
            _coefficients[index] /= _size.y;
            _coefficients[index] *= 4.0f * glm::pi<float>();
        }
    }

    cudaError_t err = cudaGetLastError();
    std::cout << cudaGetErrorString(err) << std::endl;
}

void SHSampler::reconstruct() {
    if (_reconstructed != 0) {
        glDeleteTextures(1, &_reconstructed);
    }

    glm::vec3 *texture = new glm::vec3[_size.x * _size.y];
    for (int y = 0; y < _size.y; y++) {
        for (int x = 0; x < _size.x; x++) {
            texture[y * _size.x + x] = glm::vec3(0.0f);
        }
    }

    cudaError_t err = cudaGetLastError();
    std::cout << cudaGetErrorString(err) << std::endl;

    {
        CudaPtr<glm::vec3> texture_cuda(texture, _size.x * _size.y);

        for (int i = 0; i < _coefficients.size(); i++) {
            {
                // 1. weight all SH textures
                CudaPtr<glm::vec3> sh_cuda(buffer[i], _size.x * _size.y);
                cuda::weight<<<_size.y, _size.x>>>(sh_cuda(), _coefficients[i], _size);

                // 2. sum all SH textures
                cuda::add_to<<<_size.y, _size.x>>>(texture_cuda(), sh_cuda(), _size);
            }
        }
    }
    
    glGenTextures(1, &_reconstructed);
    glBindTexture(GL_TEXTURE_2D, _reconstructed);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _size.x, _size.y, 0, GL_RGB, GL_FLOAT, &texture[0]);
    glBindTexture(GL_TEXTURE_2D, GL_NONE);

    err = cudaGetLastError();
    std::cout << cudaGetErrorString(err) << std::endl;

    delete[] texture;
}
