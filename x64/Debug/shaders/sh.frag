#version 330 core 

in mat4 shCoeffs;

uniform vec3 scene[16];

out vec4 color;

void main() {
    vec3 objColor = vec3(1.0);
    vec3 r = vec3(0.0);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            r += (objColor * shCoeffs[i][j]) * scene[i * 4 + j];
        }
    }

    color = vec4(r, 1.0);
}
