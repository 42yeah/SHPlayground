#pragma once

#include <iostream>
#include <glm/glm.hpp>
#include <glad/glad.h>


class Window;
class SHSampler;

class EnvMap {
public:
    EnvMap() : data(nullptr), _size(0, 0), texture(0) {  }

    EnvMap(const EnvMap &) = delete;

    EnvMap(std::string path);

    ~EnvMap();

    glm::ivec2 size();

    glm::vec4 operator()(int x, int y) const;

    void active_texture(GLuint index);

private:
    glm::ivec2 _size;
    glm::vec4 *data; // width * height * RGBA
    GLuint texture;

    friend class Window;
    friend class SHSampler;
};

