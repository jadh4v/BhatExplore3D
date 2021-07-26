#version 400 core

in vec3 vertexPosition;
in vec3 vertexColor;

out vec3 f_color;

uniform mat4 camera_transform;

void main(void)
{
//    gl_Position.xyz = vertexPosition_modelspace;
//    gl_Position.w = 1.0;
    gl_Position = camera_transform * vec4( vertexPosition, 1.0 );
    f_color = vertexColor;
}
