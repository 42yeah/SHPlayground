#version 330 core


layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in mat4 aSHCoeffs;

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

out mat4 shCoeffs;


void main() {
    gl_Position = perspective * view * model * vec4(aPos, 1.0);

    shCoeffs = aSHCoeffs;
}
