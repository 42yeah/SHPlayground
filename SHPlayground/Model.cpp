#include "Model.h"



Model::Model(Vertex *vertices, int num_vertices) {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * num_vertices, vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void *) (sizeof(float) * 3));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void *) (sizeof(float) * 6));

    glBindVertexArray(GL_NONE);
    valid = true;
    this->num_vertices = num_vertices;

    centroid = glm::vec3(0.0f);
    for (int i = 0; i < num_vertices; i++) {
        centroid += vertices[i].position;
        bbox.enclose(vertices[i].position);
    }

    centroid /= num_vertices;

    _has_underlying_data = true;
    this->vertices.resize(num_vertices);
    std::memcpy(&this->vertices[0], vertices, sizeof(Vertex) * num_vertices);
}

Model::~Model() {
    if (!valid) {
        return;
    }
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    if (_has_underlying_data) {
        // do something related to that data
    }
}

std::shared_ptr<Model> Model::make_triangle_model() {
    std::shared_ptr<Model> model = std::make_unique<Model>();
    glGenVertexArrays(1, &model->vao);
    glGenBuffers(1, &model->vbo);
    glBindVertexArray(model->vao);
    glBindBuffer(GL_ARRAY_BUFFER, model->vbo);

    float data[] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void *) (sizeof(float) * 3));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void *) (sizeof(float) * 6));

    glBindVertexArray(GL_NONE);
    model->valid = true;
    model->num_vertices = 3;
    model->centroid = glm::vec3(0.25f, 0.25f, 0.0f);
    model->bbox.enclose(glm::vec3(0.0f, 0.0f, 0.0f));
    model->bbox.enclose(glm::vec3(0.5f, 0.0f, 0.0f));
    model->bbox.enclose(glm::vec3(0.0f, 0.5f, 0.0f));
    return model;
}

std::shared_ptr<Model> Model::make_rect_model() {
    std::shared_ptr<Model> model = std::make_unique<Model>();
    glGenVertexArrays(1, &model->vao);
    glGenBuffers(1, &model->vbo);
    glBindVertexArray(model->vao);
    glBindBuffer(GL_ARRAY_BUFFER, model->vbo);

    float data[] = {
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void *) (sizeof(float) * 3));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void *) (sizeof(float) * 6));

    glBindVertexArray(GL_NONE);
    model->valid = true;
    model->num_vertices = 6;
    model->centroid = glm::vec3(0.0f);
    model->bbox.enclose(glm::vec3(-1.0f, -1.0f, 0.0f));
    model->bbox.enclose(glm::vec3(1.0f, -1.0f, 0.0f));
    model->bbox.enclose(glm::vec3(1.0f, 1.0f, 0.0f));
    model->bbox.enclose(glm::vec3(-1.0f, 1.0f, 0.0f));
    return model;
}

void Model::bind() {
    glBindVertexArray(vao);
}
