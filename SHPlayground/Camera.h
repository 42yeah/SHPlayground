#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


struct Camera {
public:
    Camera() : eye(0.0f), 
        yaw(0.0f), 
        pitch(0.0f), 
        fovy(glm::radians(45.0f)),
        z_near(0.01f),
        z_far(100.0f),
        speed(1.0f),
        sensitivity(0.001f),
        base_pitch(0.0f),
        base_yaw(0.0f) {  }

    Camera(glm::vec3 eye, glm::vec3 center) : eye(eye), 
        yaw(0.0f), 
        pitch(0.0f), 
        fovy(glm::radians(45.0f)),
        z_near(0.01f),
        z_far(100.0f),
        speed(1.0f),
        sensitivity(0.001f),
        base_pitch(0.0f),
        base_yaw(0.0f) {  }

    glm::mat4 look_at();
    
    glm::mat4 perspective(float aspect);

    glm::vec3 forward();

    glm::vec3 eye;

    float yaw, pitch, sensitivity;
    float speed;
    
    // perspective
    float fovy;
    float z_near, z_far;

    float base_pitch, base_yaw;
};

