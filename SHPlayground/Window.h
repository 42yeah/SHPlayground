#pragma once

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include "ResourceManager.h"


class Window {
public:
    Window() : width(0), height(0), window(nullptr), current_model(nullptr), selected(manager.end()) {  }

    ~Window();

    Window(int width, int height, std::string title);

    Window(const Window &) = delete; // no copy constructor

    Window(Window &&window) noexcept;

    bool should_close();

    void poll_event();

    void render();

    void swap_buffers();

    void render_model_using_shader(ModelPtr model, ShaderPtr shader);

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
    ModelPtr current_model;
    ShaderPtr current_shader;
    ResourcePtr selected;

    // UI system
    std::string file_dialog_chosen_path;
};
