attribute vec2 position;
attribute float density;
uniform vec3 FillColor;
varying vec4 g_color;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    g_color = vec4(FillColor, density);
}
