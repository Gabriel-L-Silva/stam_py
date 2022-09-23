#version 120
const float SQRT_2 = 1.4142135623730951;
uniform mat4 ortho;
uniform float size, orientation, linewidth, antialias;
attribute vec3 position;
varying vec2 rotation;
varying vec2 v_size;
void main (void)
{
    rotation = vec2(cos(orientation), sin(orientation));
    gl_Position = ortho * vec4(position, 1.0);
    v_size = M_SQRT_2 * size + 2.0*(linewidth + 1.5*antialias);
    gl_PointSize = v_size;
}