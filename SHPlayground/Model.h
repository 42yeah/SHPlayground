#pragma once

#include <iostream>
#include <glad/glad.h>
#include <glm/glm.hpp>


class Window;


struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 tex_coord;
};

struct BBox {
    BBox() : min(FLT_MAX, FLT_MAX, FLT_MAX), max(-FLT_MAX, -FLT_MAX, -FLT_MAX) {  }

    void enclose(glm::vec3 p) {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }

    glm::vec3 min;
    glm::vec3 max;
};


class Model {
public:
    Model() : valid(false), vao(0), vbo(0), num_vertices(0), has_underlying_data(false), centroid(0.0f) {  }

    Model(const Model &) = delete;

    Model(Vertex *vertices, int num_vertices);

    int vertex_count() const {
        return num_vertices;
    };

    ~Model();

    static std::shared_ptr<Model> make_triangle_model();

    static std::shared_ptr<Model> make_rect_model();

    void bind();

    glm::vec3 get_centroid() const {
        return centroid;
    }

    BBox bbox;

private:
    bool valid;
    GLuint vao, vbo;
    int num_vertices;
    glm::vec3 centroid;

    // has_underlying_data dictates whether the model has primitive data or not (fully inside GPU),
    // so that the model can be reconstructed again if needed.
    bool has_underlying_data;

    friend class Window;
};
