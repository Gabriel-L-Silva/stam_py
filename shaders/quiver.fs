#version 120
const float PI = 3.14159265358979323846264;
const float SQRT_2 = 1.4142135623730951;
uniform float size, linewidth, antialias;
uniform vec4 fg_color, bg_color;
varying vec2 rotation;
varying vec2 v_size;
void main()
{
    vec2 P = gl_PointCoord.xy - vec2(0.5,0.5);
    P = vec2(rotation.x*P.x - rotation.y*P.y,
    rotation.y*P.x + rotation.x*P.y);
    float distance = marker(P*v_size, size);
    gl_FragColor = outline(distance,
    linewidth, antialias, fg_color, bg_color);
}