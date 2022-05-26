#pragma once

#include <iostream>
#include <vector>
#include <variant>
#include <map>
#include <optional>
#include "Model.h"
#include "Shader.h"


template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;


struct Texture {

};

using ModelPtr = std::shared_ptr<Model>;
using TexturePtr = std::shared_ptr<Texture>;
using ShaderPtr = std::shared_ptr<Shader>;

using Resource = std::variant<
    ModelPtr, 
    TexturePtr, 
    ShaderPtr>;

using ResourcePtr = std::map<std::string, Resource>::iterator;

// resource manager handles textures & models
class ResourceManager {
public:
    ResourceManager() {  }

    ResourceManager(const ResourceManager &) = delete;

    ResourceManager(ResourceManager &&);

    ResourcePtr texture(std::string alias, std::string path);

    ResourcePtr texture(std::string alias);

    ResourcePtr model(std::string alias, std::string path);

    ResourcePtr model(std::string alias);

    ResourcePtr shader(std::string alias, std::string vertex_path, std::string fragment_path);

    ResourcePtr shader(std::string alias);

    ResourcePtr load_test_triangle_model(std::string alias);

    ResourcePtr begin() {
        return resources.begin();
    }

    ResourcePtr end() {
        return resources.end();
    }

private:
    std::map<std::string, Resource> resources;
};

