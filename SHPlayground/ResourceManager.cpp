#include "ResourceManager.h"


ResourcePtr ResourceManager::texture(std::string alias, std::string path) {

    return resources.end();
}

ResourcePtr ResourceManager::model(std::string alias, std::string path) {
    if (resources.find(alias) == resources.end()) {
        // try to load it in

    }
    return resources.end();
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
    std::shared_ptr<Model> triangle = Model::make_triangle_model();

    resources[alias] = Resource(triangle);
    return resources.find(alias);
}
