#version 120
const float M_SQRT_2 = 1.4142135623730951;
const float PI = 3.14159265358979323846264;
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
uniform float size, linewidth, antialias;
uniform sampler2D velocity;
in vec2 position;
out vec2 rotation;
out float v_size;
void main (void)
{
    rotation = texture(velocity, position/PI).xy;
    gl_Position = u_projection * u_view * u_model * vec4(position, 0, 1.0);
    v_size = M_SQRT_2 * size + 2.0*(linewidth + 1.5*antialias);
    gl_PointSize = v_size;
}