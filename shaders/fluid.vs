attribute vec2 position;
attribute float density;
uniform vec3 color;
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
varying vec4 g_color;

void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(position, 0.0, 1.0);
    g_color = vec4(color, density);
}
