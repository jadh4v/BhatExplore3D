#version 400 core

in vec3 vertexPosition;
in vec3 vertexColors;
out vec3 f_color;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
//uniform mat4 cameraTransform;
uniform mat4 projMatrix;

void main(void)
{
//    mat4 cameraTransform = modelMatrix * viewMatrix * projMatrix;
    mat4 cameraTransform = projMatrix * viewMatrix * modelMatrix;
    gl_Position = cameraTransform * vec4( vertexPosition, 1.0f );
    f_color = vertexColors;
}
