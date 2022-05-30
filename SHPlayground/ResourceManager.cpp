#include "ResourceManager.h"


ResourcePtr ResourceManager::texture(std::string alias, std::string path) {

    return resources.end();
}

ResourcePtr ResourceManager::envmap(std::string alias, std::string path) {
    if (resources.find(alias) == resources.end()) {
        std::shared_ptr<EnvMap> envmap = std::make_shared<EnvMap>(path);
        resources[alias] = envmap;
    }
    return resources.find(alias);
}

ResourcePtr ResourceManager::envmap(std::string alias) {
    return resources.find(alias);
}

ResourcePtr ResourceManager::model(std::string alias) {
    return resources.find(alias);
}

ResourcePtr ResourceManager::model(std::string alias, Vertex* vertices, int num_vertices) {
    if (resources.find(alias) == resources.end()) {
        std::shared_ptr<Model> model = std::make_shared<Model>(vertices, num_vertices);
        resources[alias] = model;
    }
    return resources.find(alias);
}

ResourcePtr ResourceManager::shader(std::string alias, std::string vertex_path, std::string fragment_path) {
    if (resources.find(alias) == resources.end()) {
        // try to load it in
        std::shared_ptr<Shader> shader = std::make_shared<Shader>(vertex_path, fragment_path);
        resources[alias] = shader;
    }
    return resources.find(alias);
}

ResourcePtr ResourceManager::shader(std::string alias) {
    return resources.find(alias);
}

ResourcePtr ResourceManager::load_test_triangle_model(std::string alias) {
    if (resources.find(alias) == resources.end()) {
        std::shared_ptr<Model> triangle = Model::make_triangle_model();

        resources[alias] = Resource(triangle);
    }
    
    return resources.find(alias);
}

ResourcePtr ResourceManager::load_test_rect_model(std::string alias) {
    if (resources.find(alias) == resources.end()) {
        std::shared_ptr<Model> rect = Model::make_rect_model();
        resources[alias] = rect;
    }
    
    return resources.find(alias);
}

ResourcePtr ResourceManager::camera(std::string alias) {
    if (resources.find(alias) == resources.end()) {
        resources[alias] = std::make_shared<Camera>(glm::vec3(0.0f, 0.2f, 1.0f), glm::vec3(0.0f));
    }
    return resources.find(alias);
}

ResourcePtr ResourceManager::scene(std::string alias, std::string path) {
    if (resources.find(alias) == resources.end()) {
        resources[alias] = std::make_shared<Scene>(*this, path);
    }
    return resources.find(alias);
}

ResourcePtr ResourceManager::scene(std::string alias) {
    if (resources.find(alias) == resources.end()) {
        resources[alias] = std::make_shared<Scene>();
    }
    return resources.find(alias);
}
