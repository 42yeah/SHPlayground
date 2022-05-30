#include "Camera.h"

glm::mat4 Camera::look_at() {
    return glm::lookAt(eye, eye + forward(), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::perspective(float aspect) {
    return glm::perspective(fovy, aspect, z_near, z_far);
}

glm::vec3 Camera::forward() {
    // default camera aims at negative Z direction
    float mod_pitch = pitch + base_pitch;
    float mod_yaw = yaw + base_yaw;
    return glm::vec3(glm::sin(mod_yaw) * glm::cos(mod_pitch), glm::sin(mod_pitch), -glm::cos(mod_pitch) * glm::cos(mod_yaw));
}
