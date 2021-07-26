#version 400 core

in vec3 f_color;
out vec4 out_color;

void main(void)
{
//    color = vec3(0,1,0);
    //gl_FragColor = vec4(f_color.x, f_color.y, f_color.z, 1.0);
    out_color = vec4(f_color.x, f_color.y, f_color.z, 1.0);
}
