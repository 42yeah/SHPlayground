#pragma once

#include <iostream>
#include <glad/glad.h>


class Window;


class Model {
public:
    Model() : valid(false), vao(0), vbo(0), num_vertices(0), has_underlying_data(false) {  }

    Model(const Model &) = delete;

    int vertex_count() const {
        return num_vertices;
    };

    ~Model();

    static std::shared_ptr<Model> make_triangle_model();

    void bind();

private:
    bool valid;
    GLuint vao, vbo;
    int num_vertices;

    // has_underlying_data dictates whether the model has primitive data or not (fully inside GPU),
    // so that the model can be reconstructed again if needed.
    bool has_underlying_data;

    friend class Window;
};
