#version 330 core

layout (points) in;
layout (line_strip, max_vertices = 6) out;
in vec4 color[];
in vec2 rotation[];
uniform mat4   u_projection;    // Projection matrix
out vec4 fcolor;

void build_arrow(vec4 position)
{    
    fcolor = color[0];
    gl_Position = u_projection *(position);
    vec2 r = rotation[0]*0.1;
    vec2 perp = 0.005*normalize(vec2(1.0,-r.x/r.y));
    EmitVertex();
    gl_Position = u_projection *(position + vec4(r, 0, 0));
    EmitVertex();
    gl_Position = u_projection *(position + vec4(perp+r,0,0));    // 2:bottom-right
    EmitVertex();
    gl_Position = u_projection *(position + vec4(1.25*r, 0, 0)); // top
    EmitVertex();
    gl_Position = u_projection *(position + vec4(-perp+r,0,0));    // 2:bottom-right
    EmitVertex();
    gl_Position = u_projection *(position + vec4(r, 0, 0));
    EmitVertex();
    EndPrimitive();
}

void main() {    
    build_arrow(gl_in[0].gl_Position);
}  