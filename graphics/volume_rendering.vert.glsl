#version 400 core

in vec3 vertexPosition;
in vec3 vertexColor;

out vec4 f_pos;
out vec4 f_color;
out vec3 light_dir;
out vec3 eye_dir;

uniform mat4 model_transform;
uniform mat4 view_transform;
uniform mat4 camera_transform;

void main(void)
{
    //// CALULCATE LIGHT INFORMATION
    // position of light in world coordinates
    vec4 light_world = vec4(0.0f,0.0f,2.0f,1.0f);
    // position of vertex in camera coordinates
    vec4 vertex_view = view_transform * model_transform * vec4( vertexPosition, 1.0f );
    // position of light in camera coordinates
    vec4 light_view = view_transform * model_transform * light_world;
    // direction vector from vertex to light position in camera coordinates.
    light_dir = vertex_view.xyz - light_world.xyz;

    eye_dir = vertex_view.xyz;
    //eye_dir = vertex_view.xyz - vec3(0,0,2);

    gl_Position = camera_transform * vec4( vertexPosition, 1.0f );
    f_pos = gl_Position;

    f_color = vec4( vertexColor, 1.0);

}
