#pragma once

#include <glm/glm.hpp>
#include <vector>
#include "Model.h"


class ResourceManager;

struct Object;

using ObjectPtr = std::vector<Object>::iterator;


struct Object {
    glm::mat4 transformation;

    std::shared_ptr<Model> model;
    std::string name;
    // maybe a list of textures?
};


// scene is made up with a list of objects
class Scene {
public:
    Scene() : valid(false) {  }

    Scene(ResourceManager &man, std::string scene_path);

    ~Scene();

    ObjectPtr begin() {
        return objects.begin();
    }

    ObjectPtr end() {
        return objects.end();
    }

    glm::vec3 centroid();

private:
    std::string scene_name_prefix(std::string what);

    bool valid;
    std::vector<Object> objects;
    std::string scene_name;
};

