#include "EnvMap.h"
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>


EnvMap::EnvMap(std::string path) {
    if (IsEXR(path.c_str())) {
        std::cerr << "ERR! Path is not EXR: " << path << std::endl;
        throw "ERR! Path is not EXR.";
    }
    const char *err = nullptr;
    int ret = LoadEXR((float **) &data, &_size.x, &_size.y, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        std::cerr << "ERR! TinyEXR encountered following error: " << err << std::endl;
        throw "ERR! TinyEXR encountered an error.";
    }

    // generate texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _size.x, _size.y, 0, GL_RGBA, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_2D, GL_NONE);
}

EnvMap::~EnvMap() {
    if (data) {
        delete[] data;
        glDeleteTextures(1, &texture);
    }
    
}

glm::ivec2 EnvMap::size() {
    return _size;
}

glm::vec4 EnvMap::operator()(int x, int y) const {
    // no need to repeat
    if (x < 0 || y < 0 || x > _size.x || y > _size.y) {
        return glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
    }
    return data[y * _size.x + x];
}

void EnvMap::active_texture(GLuint index) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glActiveTexture(index);
}
