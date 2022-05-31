#pragma once

#include <iostream>
#include <vector>
#include <variant>
#include <map>
#include <optional>
#include "Model.h"
#include "Shader.h"
#include "Camera.h"
#include "Scene.h"
#include "EnvMap.h"
#include "SHSampler.cuh"


template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;


struct Texture {

};

using ModelPtr = std::shared_ptr<Model>;
using TexturePtr = std::shared_ptr<Texture>;
using ShaderPtr = std::shared_ptr<Shader>;
using CameraPtr = std::shared_ptr<Camera>;
using ScenePtr = std::shared_ptr<Scene>;
using EnvMapPtr = std::shared_ptr<EnvMap>;
using SHSamplerPtr = std::shared_ptr<SHSampler>;

using Resource = std::variant<
    ModelPtr, 
    TexturePtr, 
    ShaderPtr,
    CameraPtr,
    ScenePtr,
    EnvMapPtr,
    SHSamplerPtr>;

using ResourcePtr = std::map<std::string, Resource>::iterator;

// resource manager handles textures & models
class ResourceManager {
public:
    ResourceManager() {  }

    ResourceManager(const ResourceManager &) = delete;

    ResourceManager(ResourceManager &&);

    ResourcePtr texture(std::string alias, std::string path);

    ResourcePtr texture(std::string alias);

    ResourcePtr envmap(std::string alias, std::string path);

    ResourcePtr envmap(std::string alias);

    ResourcePtr model(std::string alias);

    ResourcePtr model(std::string alias, Vertex *vertices, int num_vertices);

    ResourcePtr shader(std::string alias, std::string vertex_path, std::string fragment_path);

    ResourcePtr shader(std::string alias);

    ResourcePtr load_test_triangle_model(std::string alias);

    ResourcePtr load_test_rect_model(std::string alias);

    ResourcePtr camera(std::string alias);

    ResourcePtr scene(std::string alias, std::string path);

    ResourcePtr scene(std::string alias);

    ResourcePtr sh_sampler(std::string alias);

    ResourcePtr sh_sampler(std::string alias, int num_bands);


    ResourcePtr begin() {
        return resources.begin();
    }

    ResourcePtr end() {
        return resources.end();
    }

private:
    std::map<std::string, Resource> resources;
};

