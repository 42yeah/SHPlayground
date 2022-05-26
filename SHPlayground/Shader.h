#pragma once

#include <glad/glad.h>
#include <iostream>
#include <map>


class Window;

class Shader {
public:
    Shader() : valid(false), program(0), vertex_shader_path("no path"), fragment_shader_path("no path") {  }

    Shader(std::string vertex_path, std::string fragment_path);

    ~Shader();

    void use();

    GLint operator[](std::string uniform_name);

    std::string get_vertex_shader_path() {
        return vertex_shader_path;
    }

    std::string get_fragment_shader_path() {
        return fragment_shader_path;
    }

private:
    std::map<std::string, GLint> uniform_location;
    GLuint program;
    bool valid;

    std::string vertex_shader_path;
    std::string fragment_shader_path;

    friend class Window;
};
