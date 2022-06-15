#version 330 core

uniform sampler2D envmap;
uniform mat3 lookAt;
uniform float aspect;

in vec2 uv;
out vec4 color;

#define PI 3.14159265

void main() {
    vec3 camDir = vec3(aspect * (uv.x * 2.0 - 1.0), uv.y * 2.0 - 1.0, -1.0);
    camDir = normalize(lookAt * camDir);

    float theta = asin(camDir.y); // -PI/2 ~ PI/2
    float phi = acos(min(1.0, -camDir.z / sqrt(1.0 - camDir.y * camDir.y)));
    float phi2 = -phi;
    float d1 = abs(sin(phi) * cos(theta) - camDir.x);
    float d2 = abs(sin(phi2) * cos(theta) - camDir.x);
    if (d2 < d1) {
        phi = phi2;
    }
    theta = 1.0 - (theta + PI / 2.0) / PI;
    phi = (-phi + PI / 2.0) / 2.0 / PI;
    
    color = texture(envmap, vec2(phi, theta));
}
