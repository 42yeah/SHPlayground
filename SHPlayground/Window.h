#pragma once

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include "ResourceManager.h"


enum class FileLoadType {
    Unknown, Model, EnvMap
};


class Window {
public:
    ~Window();

    Window(int width, int height, std::string title);

    Window(const Window &) = delete; // no copy constructor

    Window(Window &&window) noexcept = delete;

    bool should_close();

    void poll_event();

    void render();

    void swap_buffers();

    void render_using_shader(Resource model, ShaderPtr shader, CameraPtr camera);

    void render_opengl_status();

    // update camera (if present)
    void update_camera();

    // UI functions
    void render_main_menu_bar();

    void render_file_dialog();

    void render_resources();

    void render_selected_properties();

private:
    int width, height;
    GLFWwindow *window;

    // resources
    ResourceManager manager;

    ResourcePtr current_model;
    ShaderPtr current_shader;
    CameraPtr current_camera;
    EnvMapPtr current_envmap;
    SHSamplerPtr current_sh_sampler;

    ResourcePtr selected;

    // UI system
    std::string file_dialog_chosen_path;
    FileLoadType file_load_type;
    float menu_bar_height;

    // chrono
    float last_instant;
    float delta_time;
    glm::vec2 cursor_state;
};
