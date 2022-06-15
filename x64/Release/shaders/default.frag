#version 330 core

uniform int id;
uniform vec3 camPos;

in vec3 normal;

in vec3 pos;
out vec4 color;

float rand(float x) {
    return fract(1234.0 * sin(x * 1000.0)) * 0.5 + 0.5;
}

void main() {
    // generate a random color
    float idF = float(id);

    vec3 objColor = vec3(rand(idF + 5.0), rand(idF + 1.0), rand(idF + 2.0)) * 0.7 + vec3(0.5, 0.5, 0.5);

    vec3 ambient = vec3(0.3, 0.3, 0.3);
    
    vec3 lightDir = vec3(0.0, 1.0, 0.0);
    float diffuseCoeff = clamp(dot(lightDir, normal), 0.0, 1.0);
    vec3 diffuse = vec3(1.0) * diffuseCoeff;

    color = vec4(clamp(ambient + diffuse, 0.0, 1.0) * objColor, 1.0);
}
