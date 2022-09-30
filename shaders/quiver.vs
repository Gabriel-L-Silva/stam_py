#version 330 core

uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix

in vec2 position;
in float Xvelocity;
in float Yvelocity;
in vec4 acolor;
in float vec_length;

out vec2 rotation;
out vec4 color;

void main (void)
{
    rotation = normalize(vec2(Xvelocity,Yvelocity));
    color = acolor;
    gl_Position = u_view * u_model * vec4(position, 0, 1.0);
}