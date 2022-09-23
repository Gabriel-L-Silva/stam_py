#version 120
const float PI = 3.14159265358979323846264;
const float SQRT_2 = 1.4142135623730951;
uniform float size, linewidth, antialias;
uniform vec4 fg_color, bg_color;
in vec2 rotation;
in float v_size;

out vec4 fragmentColor;

float line_distance(vec2 p, vec2 p1, vec2 p2) {
    vec2 center = (p1 + p2) * 0.5;
    float len = length(p2 - p1);
    vec2 dir = (p2 - p1) / len;
    vec2 rel_p = p - center;
    return dot(rel_p, vec2(dir.y, -dir.x));
}

vec4 outline(float distance, // Signed distance to line
float linewidth, // Stroke line width
float antialias, // Stroke antialiased area
vec4 stroke, // Stroke color
vec4 fill) // Fill color
{
float t = linewidth / 2.0 - antialias;
float signed_distance = distance;
float border_distance = abs(signed_distance) - t;
float alpha = border_distance / antialias;
alpha = exp(-alpha * alpha);
if( border_distance < 0.0 )
return stroke;
else if( signed_distance < 0.0 )
return mix(fill, stroke, sqrt(alpha));
else
return vec4(stroke.rgb, stroke.a * alpha);
}
float segment_distance(vec2 p, vec2 p1, vec2 p2) {
vec2 center = (p1 + p2) * 0.5;
float len = length(p2 - p1);
vec2 dir = (p2 - p1) / len;
vec2 rel_p = p - center;
float dist1 = abs(dot(rel_p, vec2(dir.y, -dir.x)));
float dist2 = abs(dot(rel_p, dir)) - 0.5*len;
return max(dist1, dist2);
}

float arrow_stealth(vec2 texcoord, float body, float head, float linewidth, float antialias)
{
    float w = linewidth/2.0 + antialias;
    vec2 start = -vec2(body/2.0, 0.0);
    vec2 end = +vec2(body/2.0, 0.0);
    float height = 0.5;
    // Head : 4 lines
    float d1 = line_distance(texcoord, end-head*vec2(+1.0,-height), end);
    float d2 = line_distance(texcoord, end-head*vec2(+1.0,-height),
                            end-vec2(3.0*head/4.0,0.0));
    float d3 = line_distance(texcoord, end-head*vec2(+1.0,+height), end);
    float d4 = line_distance(texcoord, end-head*vec2(+1.0,+0.5),
                            end-vec2(3.0*head/4.0,0.0));
    // Body : 1 segment
    float d5 = segment_distance(texcoord,
                                start,
                                end - vec2(linewidth,0.0));
    return min(d5, max( max(-d1, d3), - max(-d2,d4)));
}

void main()
{
    vec2 P = gl_PointCoord.xy - vec2(0.5,0.5);
    P = vec2(rotation.x*P.x - rotation.y*P.y,
            rotation.y*P.x + rotation.x*P.y);
    float distance = arrow_stealth(P*v_size, size);
    fragmentColor = outline(distance, linewidth, antialias, fg_color, bg_color);
}