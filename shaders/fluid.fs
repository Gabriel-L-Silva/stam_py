#version 330 core

in vec4 g_color;

out vec4 fragmentColor;
void main()
{
    fragmentColor = g_color;
}
