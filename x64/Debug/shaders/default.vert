#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

out vec3 pos;
out vec3 normal;

void main() {
    gl_Position = perspective * view * model * vec4(aPos, 1.0);
    pos = vec3(model * vec4(aPos, 1.0));
    normal = vec3(model * vec4(aNormal, 0.0));
}
