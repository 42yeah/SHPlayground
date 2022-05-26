#include "Model.h"

Model get_triangle_model() {
    Model model;

    
    return Model();
}


Model::~Model() {
    if (!valid) {
        return;
    }
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    if (has_underlying_data) {
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
        0.0f, 0.0f, 0.0f,
        0.5f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.0f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr);
    glBindVertexArray(GL_NONE);
    model->valid = true;
    model->num_vertices = 3;
    return model;
}

void Model::bind() {
    glBindVertexArray(vao);
}
