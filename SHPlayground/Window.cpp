#include "Window.h"
#include <imgui_impl_opengl3.h>
#include <imgui_impl_glfw.h>
#include <ImGuiFileDialog.h>


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
    current_model = nullptr;
    selected = manager.end();
    manager.shader("default", "shaders/default.vert", "shaders/default.frag");
    
}

Window::Window(Window &&window) noexcept {
    this->window = window.window;
    width = window.width;
    height = window.height;
    window.window = nullptr;
}

bool Window::should_close() {
    return glfwWindowShouldClose(window);
}

void Window::poll_event() {
    glfwPollEvents();
}

void Window::render() {
    glClearColor(0.2f, 0.0f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (current_model && current_shader) {
        render_model_using_shader(current_model, current_shader);
    }

    // UI stuffs
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::ShowDemoWindow();

    render_resources();

    render_selected_properties();

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

void Window::render_model_using_shader(ModelPtr model, ShaderPtr shader) {
    shader->use();
    model->bind();
    glDrawArrays(GL_TRIANGLES, 0, model->vertex_count());
}




void Window::render_main_menu_bar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open")) {
                // TODO: stop using Scotty3D's stuffs
                ImGuiFileDialog::Instance()->OpenDialog("choose_model", "Choose .obj file", ".dae,.obj", "F:\\code\\Scotty3D\\media\\");
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void Window::render_file_dialog() {
    ImGui::SetNextWindowSize({ 800, 600 }, ImGuiCond_FirstUseEver);

    if (ImGuiFileDialog::Instance()->Display("choose_model")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string path_name = ImGuiFileDialog::Instance()->GetFilePathName();
            std::string current_path = ImGuiFileDialog::Instance()->GetCurrentPath(); (void) current_path;

            file_dialog_chosen_path = path_name;

            manager.load_test_triangle_model(path_name);
            
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
                        str = "[T] " + it->first;
                    },
                    [&](ShaderPtr s) {
                        str = "[S] " + it->first;
                    },
                    [&](ModelPtr m) {
                        str = "[M] " + it->first;
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

    if (ImGui::Begin("Properties")) {
        std::visit(overloaded {
            [&](ModelPtr m) {
                std::string title = "[MODEL] " + selected->first;
                ImGui::Text(title.c_str());
                ImGui::Separator();
                ImGui::Text("PROPERTIES");
                ImGui::Text("#vertices: %d", m->vertex_count());
                ImGui::Separator();
                ImGui::Text("REFLECTION");
                ImGui::Text("OpenGL vertex array object: %u", m->vao);
                ImGui::Text("OpenGL vertex buffer object: %u", m->vbo);
                if (m->has_underlying_data) {
                    ImGui::Text("Has underlying data: yes");
                } else {
                    ImGui::Text("Has underlying data: no");
                }

                if (current_model != m && ImGui::Button("Define as current model")) {
                    current_model = m;
                }
            },
            [&](ShaderPtr s) {
                std::string title = "[MODEL] " + selected->first;
                ImGui::Text(title.c_str());
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

                if (current_shader != s && ImGui::Button("Define as current shader")) {
                    current_shader = s;
                }
            },
            [&](auto h) {
                ImGui::Text("[UNKNOWN] %s", selected->first.c_str());
            }
        }, selected->second);
    }
    ImGui::End();
}
