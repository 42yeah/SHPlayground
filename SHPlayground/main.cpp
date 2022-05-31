#include <iostream>
#include "Window.h"
#include <GLFW/glfw3.h>


int main() {
    Window window(1280 * 2, 720 * 2, "SH Playground");

    while (!window.should_close()) {
        window.poll_event();
        window.render();
        window.swap_buffers();
    }

    return 0;
}
