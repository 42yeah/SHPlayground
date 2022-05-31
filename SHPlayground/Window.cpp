#include "Window.h"
#include <imgui_impl_opengl3.h>
#include <imgui_impl_glfw.h>
#include <ImGuiFileDialog.h>
#include <glm/gtc/type_ptr.hpp>


Window::~Window() {
    if (window) {
        glfwDestroyWindow(window);
    }

}

Window::Window(int width, int height, std::string title) : width(width), height(height) {
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    glfwMakeContextCurrent(window);

    gladLoadGL();

    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // initialize ImGui
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
    io.IniFilename = nullptr;
    io.FontGlobalScale = 2.5f;

    // resource manager
    current_model = manager.end();;
    selected = manager.end();
    manager.shader("default shader", "shaders/default.vert", "shaders/default.frag");
    manager.camera("default camera");

    // chrono
    last_instant = glfwGetTime();
    delta_time = 0.0f;
    cursor_state = glm::vec2(-1.0f, -1.0f);
}

bool Window::should_close() {
    return glfwWindowShouldClose(window);
}

void Window::poll_event() {
    glfwPollEvents();
}

void Window::render() {
    float now = glfwGetTime();
    delta_time = now - last_instant;
    last_instant = now;

    glClearColor(0.2f, 0.0f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    update_camera();

    if (current_model != manager.end() && current_shader && current_camera) {
        render_using_shader(current_model->second, current_shader, current_camera);
    }

    // UI stuffs
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ImGui::ShowDemoWindow();

    render_resources();

    render_selected_properties();

    render_opengl_status();

    render_main_menu_bar();

    render_file_dialog();
    
    ImGui::Render();
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Window::swap_buffers() {
    glfwSwapBuffers(window);
}

void Window::render_using_shader(Resource res, ShaderPtr shader_ptr, CameraPtr camera) {
    if (current_envmap) {
        // render environment map first.
        glDisable(GL_DEPTH_TEST);
        
        ShaderPtr envmap_shader_ptr = std::get<ShaderPtr>(manager.shader("envmap shader", "shaders/envmap.vert", "shaders/envmap.frag")->second);
        Shader &envmap_shader = *envmap_shader_ptr;
        
        ModelPtr rect = std::get<ModelPtr>(manager.load_test_rect_model("envmap rect")->second);

        envmap_shader.use();
        current_envmap->active_texture(GL_TEXTURE0);
        glUniform1i(envmap_shader["envmap"], 0);
        glUniform1f(envmap_shader["aspect"], (float) width / height);
        glm::mat3 look_at = glm::transpose(camera->look_at());
        glUniformMatrix3fv(envmap_shader["lookAt"], 1, GL_FALSE, glm::value_ptr(look_at));

        rect->bind();
        glDrawArrays(GL_TRIANGLES, 0, rect->vertex_count());

        glEnable(GL_DEPTH_TEST);
    }

    Shader &shader = *shader_ptr;
    shader.use();
    glm::mat4 model_mat(1.0f);

    glUniformMatrix4fv(shader["model"], 1, GL_FALSE, glm::value_ptr(model_mat));
    glUniformMatrix4fv(shader["view"], 1, GL_FALSE, glm::value_ptr(camera->look_at()));
    glUniformMatrix4fv(shader["perspective"], 1, GL_FALSE, glm::value_ptr(camera->perspective((float) width / height)));
    glUniform1i(shader["id"], 0);

    glUniform3f(shader["camPos"], camera->eye.x, camera->eye.y, camera->eye.z);

    std::visit(overloaded {
        [&](ModelPtr m) {
            m->bind();
            glDrawArrays(GL_TRIANGLES, 0, m->vertex_count());
        },
        [&](ScenePtr s) {
            int id = 0;
            for (auto it = s->begin(); it != s->end(); it++) {
                glUniform1i(shader["id"], id);
                glUniformMatrix4fv(shader["model"], 1, GL_FALSE, glm::value_ptr(it->transformation));
                it->model->bind();
                glDrawArrays(GL_TRIANGLES, 0, it->model->vertex_count());
                id++;
            }
        },
        [&](auto a) {
            throw "Unsupported render type";
        }
    }, res);
}


void Window::update_camera() {
    if (!current_camera) {
        return;
    }

    Camera &cam = *current_camera;
    glm::vec3 forward = cam.forward();
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    if (glfwGetKey(window, GLFW_KEY_W)) {
        cam.eye += cam.speed * forward * delta_time;
    }
    if (glfwGetKey(window, GLFW_KEY_A)) {
        cam.eye -= cam.speed * right * delta_time;
    }
    if (glfwGetKey(window, GLFW_KEY_S)) {
        cam.eye -= cam.speed * forward * delta_time;
    }
    if (glfwGetKey(window, GLFW_KEY_D)) {
        cam.eye += cam.speed * right * delta_time;
    }

    int mouse_button = glfwGetMouseButton(window, 0);
    if (mouse_button == GLFW_RELEASE) {
        // finalize camera
        cursor_state = glm::vec2(-1.0f, -1.0f);
        cam.base_pitch += cam.pitch;
        cam.base_yaw += cam.yaw;
        cam.pitch = 0.0f;
        cam.yaw = 0.0f;
    } else if (mouse_button == GLFW_PRESS && cursor_state.x < 0.0f) {
        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);
        cursor_state = glm::vec2(mouse_x, mouse_y);
    } else {
        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);

        cam.yaw = (mouse_x - cursor_state.x) * cam.sensitivity;
        cam.pitch = (cursor_state.y - mouse_y) * cam.sensitivity;

        if (cam.pitch + cam.base_pitch >= glm::pi<float>() / 2.0f - 0.001f) {
            cam.pitch = glm::pi<float>() / 2.0f - cam.base_pitch - 0.001f;
        }
        if (cam.pitch + cam.base_pitch <= -glm::pi<float>() / 2.0f + 0.001f) {
            cam.pitch = -glm::pi<float>() / 2.0f - cam.base_pitch + 0.001f;
        }

    }

}




void Window::render_main_menu_bar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open")) {
                // TODO: stop using Scotty3D's stuffs
                file_load_type = FileLoadType::Model;
                ImGuiFileDialog::Instance()->OpenDialog("choose", "Choose .obj file", ".dae,.obj", "F:\\code\\Scotty3D\\media\\");
            }
            
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            if (ImGui::MenuItem("Load EnvMap")) {
                file_load_type = FileLoadType::EnvMap;
                ImGuiFileDialog::Instance()->OpenDialog("choose", "Choose .exr file", ".exr", "F:\\academia\\exr\\");
            }
            if (ImGui::MenuItem("Create SH sample")) {
                // I am too lazy; just create one
                
                if (manager.sh_sampler("SH sampler") == manager.end()) {
                    manager.sh_sampler("SH sampler", 0);
                } else {
                    int i = 1;
                    while (manager.sh_sampler("SH sampler " + std::to_string(i)) != manager.end()) {
                        i++;
                    }
                    manager.sh_sampler("SH sampler " + std::to_string(i), 0);
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Debug")) {
            if (ImGui::MenuItem("Load in test triangle")) {
                manager.load_test_triangle_model("test triangle");
            }
            if (ImGui::MenuItem("Load in test rect")) {
                manager.load_test_rect_model("test rect");
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void Window::render_file_dialog() {
    ImGui::SetNextWindowSize({ 800, 600 }, ImGuiCond_FirstUseEver);

    if (ImGuiFileDialog::Instance()->Display("choose")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string path_name = ImGuiFileDialog::Instance()->GetFilePathName();
            std::string current_path = ImGuiFileDialog::Instance()->GetCurrentPath(); (void) current_path;
            file_dialog_chosen_path = path_name;

            switch (file_load_type) {
            case FileLoadType::Unknown:
                throw "ERR! Unknown file load type.";
                break;

            case FileLoadType::Model:
                manager.scene(path_name, path_name);
                break;

            case FileLoadType::EnvMap:
                manager.envmap(path_name, path_name);
                break;
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }
}

void Window::render_resources() {
    ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos({ 0, 40 }, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Resources")) {
        if (ImGui::BeginListBox("##resources", { 500, 800 })) {
            for (ResourcePtr it = manager.begin(); it != manager.end(); it++) {
                auto second = it->second;
                bool current = selected == it;
                std::string str = "";
                std::visit(overloaded {
                    [&](TexturePtr t) {
                        str = "[T]  " + it->first;
                    },
                    [&](ShaderPtr s) {
                        str = "[S]  " + it->first;
                    },
                    [&](ModelPtr m) {
                        str = "[M]  " + it->first;
                    },
                    [&](CameraPtr c) {
                        str = "[C]  " + it->first;
                    },
                    [&](ScenePtr s) {
                        str = "[Sc] " + it->first;
                    },
                    [&](EnvMapPtr e) {
                        str = "[E]  " + it->first;
                    },
                    [&](SHSamplerPtr s) {
                        str = "[SH] " + it->first;
                    }
                }, second);

                if (ImGui::Selectable(str.c_str(), &current)) {
                    selected = it;
                }
            }
            ImGui::EndListBox();
        }
    }
    ImGui::End();
}

void Window::render_selected_properties() {
    if (selected == manager.end()) {
        return;
    }

    ImGui::SetNextWindowSize({ 800, 600 }, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos({ 0, 840 }, ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Properties")) {
        std::visit(overloaded {
            [&](ModelPtr m) {
                std::string title = "[MODEL] " + selected->first;
                ImGui::Text(title.c_str());
                if (current_model != selected && ImGui::Button("Define as current model")) {
                    current_model = selected;
                }
                ImGui::Separator();
                ImGui::Text("PROPERTIES");
                ImGui::Text("#vertices: %d", m->vertex_count());
                ImGui::Text("centroid: (%f, %f, %f)", m->get_centroid().x, m->get_centroid().y, m->get_centroid().z);
                ImGui::Text("bounding box min: (%f, %f, %f)", m->bbox.min.x, m->bbox.min.y, m->bbox.min.z);
                ImGui::Text("bounding box max: (%f, %f, %f)", m->bbox.max.x, m->bbox.max.y, m->bbox.max.z);
                ImGui::Separator();
                ImGui::Text("REFLECTION");
                ImGui::Text("OpenGL vertex array object: %u", m->vao);
                ImGui::Text("OpenGL vertex buffer object: %u", m->vbo);
                if (m->has_underlying_data) {
                    ImGui::Text("Has underlying data: yes");
                } else {
                    ImGui::Text("Has underlying data: no");
                }

            },
            [&](ShaderPtr s) {
                std::string title = "[SHADER] " + selected->first;
                ImGui::Text(title.c_str());
                if (current_shader != s && ImGui::Button("Define as current shader")) {
                    current_shader = s;
                }
                ImGui::Separator();
                ImGui::Text("PROPERTIES");
                ImGui::Text("Vertex shader path: %s", s->get_vertex_shader_path().c_str());
                ImGui::Text("Fragment shader path: %s", s->get_fragment_shader_path().c_str());
                ImGui::Separator();
                ImGui::Text("REFLECTION");
                ImGui::Text("OpenGL program ID: %u", s->program);
                ImGui::Text("Cached uniform locations: %u", s->uniform_location.size());
                if (ImGui::BeginListBox("uniforms")) {
                    for (auto it = s->uniform_location.begin(); it != s->uniform_location.end(); it++) {
                        std::string label = it->first + ": " + std::to_string(it->second);
                        ImGui::Selectable(label.c_str());
                    }
                    ImGui::EndListBox();
                }
            },
            [&](CameraPtr c) {
                ImGui::Text("[CAMERA] %s", selected->first.c_str());
                if (current_camera != c && ImGui::Button("Define as current camera")) {
                    current_camera = c;
                }
                ImGui::Separator();
                ImGui::Text("PROPERTIES");
                ImGui::Text("Eye: (%f, %f, %f)", c->eye.x, c->eye.y, c->eye.z);
                glm::vec3 forward = c->forward();
                ImGui::Text("Forward: (%f, %f, %f)", forward.x, forward.y, forward.z);
                ImGui::Text("Speed: %f", c->speed);
                ImGui::Text("PY: (%f, %f)", c->pitch, c->yaw);
                ImGui::Text("Base PY: (%f, %f)", c->base_pitch, c->base_yaw);
                ImGui::Text("Sensitivity: %f", c->sensitivity);
                ImGui::Text("Z near & far: %f %f", c->z_near, c->z_far);
                ImGui::Text("fovy: %f", c->fovy);

                glm::mat4 look_at = c->look_at();
                glm::mat4 perspective = c->perspective((float) width / height);
                ImGui::Text("Camera view matrix:");
                ImGui::BeginTable("cam_lookat", 4);
                
                for (int r = 0; r < 4; r++) {
                    ImGui::TableNextRow();
                    for (int c = 0; c < 4; c++) {
                        std::string matid = "##cam_lookat_" + std::to_string(r) + "_" + std::to_string(c);
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth(-FLT_MIN);
                        ImGui::InputFloat(matid.c_str(), &look_at[c][r]);
                    }
                }
                ImGui::EndTable();

                ImGui::Text("Camera perspective matrix:");
                ImGui::BeginTable("cam_perspective", 4);

                for (int r = 0; r < 4; r++) {
                    ImGui::TableNextRow();
                    for (int c = 0; c < 4; c++) {
                        std::string matid = "##cam_perspective" + std::to_string(r) + "_" + std::to_string(c);
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth(-FLT_MIN);
                        ImGui::InputFloat(matid.c_str(), &perspective[c][r]);
                    }
                }
                ImGui::EndTable();
            },
            [&](ScenePtr s) {
                ImGui::Text("[SCENE] %s", selected->first.c_str());
                if (current_model != selected && ImGui::Button("Define as current scene")) {
                    current_model = selected;
                }
                ImGui::Separator();
                ImGui::Text("PROPERTIES");
                int dist = std::distance(s->begin(), s->end());
                ImGui::Text("#objects: %d", dist);

                glm::vec3 centroid = s->centroid();
                ImGui::Text("centroid: (%f, %f, %f)", centroid.x, centroid.y, centroid.z);

                if (ImGui::BeginListBox("objects")) {
                    for (auto it = s->begin(); it != s->end(); it++) {
                        ImGui::Selectable(it->name.c_str());
                    }
                    ImGui::EndListBox();
                }
            },
            [&](EnvMapPtr e) {
                // we are using static... oh no!
                static glm::ivec2 sample(0, 0);
                ImGui::Text("[ENVMAP] %s", selected->first.c_str());
                if (current_envmap != e && ImGui::Button("Define as current envmap")) {
                    current_envmap = e;
                }
                ImGui::Separator();

                ImGui::Text("PROPERTIES");
                ImGui::Text("size: (%d, %d)", e->size().x, e->size().y);

                ImGui::Separator();
                ImGui::Text("REFLECTION");
                ImGui::Text("OpenGL texture ID: %u", e->texture);
                ImGui::Text("Sample: "); ImGui::SameLine();
                ImGui::InputInt("##sample_x", &sample.x); ImGui::SameLine();
                ImGui::InputInt("##sample_y", &sample.y);

                glm::vec4 color = e->operator()(sample.x, sample.y);
                ImGui::ColorEdit4("sample color", (float *) &color);
                
                float aspect = e->size().x / e->size().y;
                float w = ImGui::GetWindowWidth();
                ImGui::Image((void *) e->texture, ImVec2(w, w / aspect));
            },
            [&](SHSamplerPtr s) {
                static int num_bands = 4, size_width = 128, size_height = 128;
                static int l = 0, m = 0;
                ImGui::Text("[SHSAMPLER] %s", selected->first.c_str());
                ImGui::Separator();
                ImGui::Text("PROPERTIES");
                ImGui::Text("size: (%d, %d)", s->size().x, s->size().y);
                ImGui::Text("#bands: %d", s->num_bands());
                ImGui::Separator();
                ImGui::Text("RECONFIGURE");
                ImGui::InputInt("new band level", &num_bands);

                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::BeginTable("##resize_sh", 2);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputInt("##width", &size_width);
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputInt("##height", &size_height);
                ImGui::EndTable();
                if (ImGui::Button("Reconfigure")) {
                    s->resize(glm::ivec2(size_width, size_height), num_bands);
                    s->visualize();
                }
                ImGui::SameLine();
                if (ImGui::Button("Calculate SH")) {
                    s->sample_sh();
                    s->visualize();
                }
                if (current_envmap && ImGui::Button("Calculate envlight coefficients")) {
                    s->calc_coefficients(*current_envmap);
                }

                if (s->coefficients().size() > 0 && s->textures()) {
                    if (ImGui::Button("Reconstruct")) {
                        s->reconstruct();
                    }
                    for (int l = 0; l < s->num_bands(); l++) {
                        for (int m = -l; m <= l; m++) {
                            int index = l * (l + 1) + m;
                            glm::vec3 coeff = s->coefficients()[index] * 0.5f + 0.5f;

                            ImGui::ColorEdit3((std::string("coefficient ") + std::to_string(index)).c_str(), &coeff[0]);
                        }
                    }
                }
                if (s->reconstructed() > 0) {
                    ImGui::Text("Reconstructed");
                    ImGui::Image((void *) s->reconstructed(), ImVec2(s->size().x, s->size().y));
                }
                if (s->textures()) {
                    ImGui::SliderInt("l", &l, 0, s->num_bands() - 1);
                    ImGui::SliderInt("m", &m, -l, l);
                    l = glm::min<int>(l, s->num_bands() - 1);
                    m = glm::clamp<int>(m, -l, l);
                    int index = l * (l + 1) + m;
                    ImGui::Text("Index: %d", index);
                    ImGui::Text("OpenGL texture ID: %u", s->textures()[index]);
                    ImGui::Image((void *) s->textures()[index], ImVec2(s->size().x, s->size().y));
                }
            },
            [&](auto h) {
                ImGui::Text("[UNKNOWN] %s", selected->first.c_str());
            }
        }, selected->second);
    }
    ImGui::End();
}

void Window::render_opengl_status() {
    ImGui::SetNextWindowSize({ 500, 100 }, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos({ 500, 40 }, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("OpenGL status")) {
        GLuint error = glGetError();
        ImGui::Text("OpenGL error code: %u", error);
    }
    ImGui::End();
}
