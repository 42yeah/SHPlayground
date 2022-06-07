#pragma once

#include <iostream>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include "Vertex.h"


class Window;


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
    Model() : valid(false), vao(0), vbo(0), num_vertices(0), _has_underlying_data(false), centroid(0.0f) {  }

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

    inline bool has_underlying_data() const {
        return _has_underlying_data;
    }

    const inline std::vector<Vertex> &get_vertices() const {
        return vertices;
    }

    BBox bbox;

private:
    bool valid;
    GLuint vao, vbo;
    int num_vertices;
    glm::vec3 centroid;

    // has_underlying_data dictates whether the model has primitive data or not (fully inside GPU),
    // so that the model can be reconstructed again if needed.
    bool _has_underlying_data;
    std::vector<Vertex> vertices;

    friend class Window;
    friend class SHSampler;
};
