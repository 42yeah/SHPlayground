#include "SHSampler.cuh"
#include <fstream>
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

    cudaError_t err = cudaGetLastError();
    std::cout << cudaGetErrorString(err) << std::endl;

    delete[] texture;
}

void SHSampler::evaluate_scene_coeffs(std::shared_ptr<Scene> scene) {
    // just need to evaluate all models in scenes once
    std::vector<std::shared_ptr<Model> > unique_models;

    for (auto it = scene->begin(); it != scene->end(); it++) {
        if (std::find(unique_models.begin(), unique_models.end(), it->model) == unique_models.end()) {
            unique_models.push_back(it->model);
        }
    }

    for (auto it = unique_models.begin(); it != unique_models.end(); it++) {
        evaluate_model_coeffs(*it);
    }
}

void SHSampler::evaluate_model_coeffs(std::shared_ptr<Model> model) {
    if (!model->has_underlying_data() || !model->valid) {
        return;
    }
    const std::vector<Vertex> &vertices = model->get_vertices();

    // perform SH projection
    std::vector<glm::vec3> vertices_coeffs;
    // _coefficients.size() for EACH vertex
    vertices_coeffs.resize(_coefficients.size() * vertices.size());
    
    std::vector<glm::vec3> result_vertical;
    result_vertical.resize(_size.y);

    // for each SH function...
    for (int i = 0; i < _coefficients.size(); i++) {

        CudaPtr<glm::vec3> sh(buffer[i], _size.x * _size.y);

        // for each vertex...
        for (int j = 0; j < vertices.size(); j++) {
            // get the resulting vector...
            glm::vec3 &result = vertices_coeffs[j * _coefficients.size() + i];
            result = glm::vec3(0.0f);

            // zero out vertical results...
            for (int k = 0; k < result_vertical.size(); k++) {
                result_vertical[k] = glm::vec3(0.0f);
            }

            {
                CudaPtr<glm::vec3> result_vertical_cuda(&result_vertical[0], _size.y);

                // perform SH projection...
                cuda::vertex_sh_project<<<1, _size.y>>>(sh(), vertices[j], _size, result_vertical_cuda());
            }

            // sum up the result...
            for (int k = 0; k < result_vertical.size(); k++) {
                result += result_vertical[k];
            }
            result /= _size.y;
            result *= 4.0f * glm::pi<float>();
        }

    }

    // for now, all three channels are the same - we are just going to take the first

    cudaError_t err = cudaGetLastError();
    std::cout << cudaGetErrorString(err) << std::endl;

    // reset model VAO and VBO
    glDeleteVertexArrays(1, &model->vao);
    glDeleteBuffers(1, &model->vbo);

    std::vector<float> vertex_data;
    
    // number of floats: vertices(3) + normals(3) + texCoords(2) + coeffs(n)
    constexpr int offset = 3 + 3 + 2;
    int n = _coefficients.size();
    vertex_data.resize(vertices.size() * (offset + n));

    for (int i = 0; i < vertices.size(); i++) {
        std::memcpy(&vertex_data[i * (offset + n)], &vertices[i], sizeof(Vertex));

        for (int j = 0; j < n; j++) {
            vertex_data[i * (offset + n) + offset + j] = vertices_coeffs[i * n + j].x;
        }
    }

    glGenVertexArrays(1, &model->vao);
    std::cout << glGetError() << std::endl;
    glGenBuffers(1, &model->vbo);
    std::cout << glGetError() << std::endl;
    glBindVertexArray(model->vao);
    std::cout << glGetError() << std::endl;
    glBindBuffer(GL_ARRAY_BUFFER, model->vbo);
    std::cout << glGetError() << std::endl;
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_data.size(), &vertex_data[0], GL_STATIC_DRAW);
    std::cout << glGetError() << std::endl;
    glEnableVertexAttribArray(0);
    std::cout << glGetError() << std::endl;
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * (offset + n), nullptr);
    std::cout << glGetError() << std::endl;
    glEnableVertexAttribArray(1);
    std::cout << glGetError() << std::endl;
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * (offset + n), (void *) (sizeof(float) * 3));
    std::cout << glGetError() << std::endl;
    glEnableVertexAttribArray(2);
    std::cout << glGetError() << std::endl;
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * (offset + n), (void *) (sizeof(float) * 6));
    std::cout << glGetError() << std::endl;
    glEnableVertexAttribArray(3);
    std::cout << glGetError() << std::endl;
    glVertexAttribPointer(3, n / 4, GL_FLOAT, GL_FALSE, sizeof(float) * (offset + n), (void *) (sizeof(float) * 8));
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, n / 4, GL_FLOAT, GL_FALSE, sizeof(float) * (offset + n), (void *) (sizeof(float) * 12));
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, n / 4, GL_FLOAT, GL_FALSE, sizeof(float) * (offset + n), (void *) (sizeof(float) * 16));
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, n / 4, GL_FLOAT, GL_FALSE, sizeof(float) * (offset + n), (void *) (sizeof(float) * 20));

    std::cout << glGetError() << std::endl;
    glBindVertexArray(GL_NONE);
    std::cout << glGetError() << std::endl;

    if (std::strlen(vertex_coeff_export_path) > 0) {
        std::ofstream writer(vertex_coeff_export_path);
        if (!writer.good()) {
            std::cerr << "WARNING! Cannot export CSV: bad writer." << std::endl;
            return;
        }

        // ;-separated CSV
        writer << "x;y;z;nx;ny;nz;tu;tv;0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15" << std::endl;
        for (int i = 0; i < vertices.size(); i++) {
            const Vertex &v = vertices[i];
            writer << v.position.x << ";" << v.position.y << ";" << v.position.z << ";" << v.normal.x << ";" << v.normal.y << ";" << v.normal.z << ";" << v.tex_coord.x << ";" << v.tex_coord.y;
            for (int j = 0; j < n; j++) {
                writer << ";" << vertices_coeffs[i * n + j].x;
            }
            writer << std::endl;
        }
        writer.close();
    }
}
