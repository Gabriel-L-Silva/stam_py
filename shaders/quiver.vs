#version 120
const float M_SQRT_2 = 1.4142135623730951;
const float PI = 3.14159265358979323846264;
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix

layout(location = 0) in vec2 velocity;
layout(location = 1) in vec2 position;
in vec4 acolor;
in float vec_length;

out vec2 rotation;
out vec4 color;

void main (void)
{
    rotation = normalize(velocity);
    color = acolor;
    gl_Position = u_view * u_model * vec4(position, 0, 1.0);
}