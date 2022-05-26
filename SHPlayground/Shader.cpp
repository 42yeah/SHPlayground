#include "Shader.h"
#include <fstream>
#include <sstream>
#include <optional>


std::optional<GLuint> compile(GLuint type, std::string path) {
    std::ifstream reader(path);
    if (!reader.good()) {
        std::cerr << "Cannot load path: " << path << std::endl;
        return std::nullopt;
    }
    std::stringstream stream;
    stream << reader.rdbuf();
    std::string str = stream.str();
    const char *raw = str.c_str();

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &raw, nullptr);
    glCompileShader(shader);
    
    GLint compile_status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);

    if (compile_status == GL_FALSE) {
        char log[512] = { 0 };
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "FATAL! Failed to compile shader: " << log << std::endl;
        return std::nullopt;
    }
    return shader;
}

Shader::Shader(std::string vertex_path, std::string fragment_path) : vertex_shader_path(vertex_path), fragment_shader_path(fragment_path) {
    program = glCreateProgram();
    glAttachShader(program, *compile(GL_VERTEX_SHADER, vertex_path));
    glAttachShader(program, *compile(GL_FRAGMENT_SHADER, fragment_path));
    glLinkProgram(program);

    GLint link_status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if (link_status == GL_FALSE) {
        char log[512] = { 0 };
        glGetShaderInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "FATAL! Failed to link program: " << log << std::endl;
        // just throw
        throw "FATAL! Failed to link program.";
    }
    valid = true;
}

Shader::~Shader() {
    if (!valid) {
        return;
    }
    glDeleteProgram(program);
}

void Shader::use() {
    glUseProgram(program);
}

GLint Shader::operator[](std::string uniform_name) {
    if (uniform_location.find(uniform_name) == uniform_location.end()) {
        GLint loc = glGetUniformLocation(program, uniform_name.c_str());
        if (loc < 0) {
            std::cerr << "WARNING! " << uniform_name << " is not found in program." << std::endl;
        }
        uniform_location[uniform_name] = loc;
    }
    return uniform_location[uniform_name];
}

