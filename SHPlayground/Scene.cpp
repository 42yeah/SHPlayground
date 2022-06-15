#include "Scene.h"
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "ResourceManager.h"


Scene::Scene(ResourceManager &man, std::string scene_path) {
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(scene_path, aiProcess_GenNormals | aiProcess_Triangulate);

    if (!scene) {
        std::cerr << "ERROR! Cannot load scene: " << scene_path << std::endl;
        throw "ERROR! Cannot load scene.";
    }

    scene_name = scene->mName.C_Str();
    
    // load in meshes (which we call models)
    for (int i = 0; i < scene->mNumMeshes; i++) {
        const aiMesh *mesh = scene->mMeshes[i];
        std::vector<Vertex> vertices;

        for (int j = 0; j < mesh->mNumVertices; j++) {
            const aiVector3D pos = mesh->mVertices[j];

            aiVector3D normal(0.0f, 0.0f, 0.0f);
            if (mesh->HasNormals()) {
                normal = mesh->mNormals[j];
            }

            aiVector3D tex_coord(0.0f, 0.0f, 0.0f);
            if (mesh->HasTextureCoords(0)) {
                tex_coord = mesh->mTextureCoords[0][j];
            }

            Vertex vert = {
                glm::vec3(pos.x, pos.y, pos.z),
                glm::vec3(normal.x, normal.y, normal.z),
                glm::vec2(tex_coord.x, tex_coord.y)
            };
            vertices.push_back(vert);
        }

        man.model(scene_name_prefix(mesh->mName.C_Str()), &vertices[0], vertices.size());
    }

    std::vector<aiNode *> to_process{ scene->mRootNode };
    while (!to_process.empty()) {
        aiNode *node = to_process.back();
        to_process.pop_back();

        for (int i = 0; i < node->mNumChildren; i++) {
            to_process.push_back(node->mChildren[i]);
        }
        
        // TODO: actually perform the transformation
        for (int i = 0; i < node->mNumMeshes; i++) {
            Object object;
            glm::mat4 transformation;
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    transformation[c][r] = node->mTransformation[r][c];
                }
            }

            object.transformation = transformation;
            std::string ref_name = scene_name_prefix(scene->mMeshes[node->mMeshes[i]]->mName.C_Str());
            object.model = std::get<ModelPtr>(man.model(ref_name)->second);
            object.name = std::string("obj_ref_") + ref_name;
            objects.push_back(object);
        }
    }

    valid = true;
}

Scene::~Scene() {
    if (!valid) {
        return;
    }
}

glm::vec3 Scene::centroid() {
    // calculate all transformed centroids
    glm::vec3 ret(0.0f);
    for (auto it = begin(); it != end(); it++) {
        ret += glm::vec3(it->transformation * glm::vec4(it->model->get_centroid(), 1.0f));
    }

    return ret;
}

std::string Scene::scene_name_prefix(std::string what) {
    return scene_name + "_" + what;
}
