#include<stdio.h>
#include<iostream>
#include<ctime>

#include<QFile>
#include<QMouseEvent>

//#define GLM_FORCE_RADIANS
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

#include <QGLFramebufferObject>
#include <QOpenGLFramebufferObject>

#include "ds/Grid.h"
//#include "vpcommon/vpVolumeRenderer.h"
#include "VolumeRenderer.h"
#include "core/macros.h"
#include "vpcommon/vpInlines.h"
#include "vpcommon/vpStructures.h"
#include "vpcommon/vpTypes.h"

using sjDS::Grid;

#define USE_GRADIENTS
GLfloat hseg_box_verts[] = {
    // Back
    0, 0, 0,
    0, 1, 0,
    1, 1, 0,

    0, 0, 0,
    1, 1, 0,
    1, 0, 0,

    // Front
    0, 0, 1,
    1, 0, 1,
    1, 1, 1,

    0, 0, 1,
    1, 1, 1,
    0, 1, 1,

    // Top
    0, 1, 0,
    0, 1, 1,
    1, 1, 1,

    0, 1, 0,
    1, 1, 1,
    1, 1, 0,

    // Bottom
    0, 0, 0,
    1, 0, 0,
    1, 0, 1,

    0, 0, 0,
    1, 0, 1,
    0, 0, 1,

    // Left
    0, 0, 0,
    0, 0, 1,
    0, 1, 1,

    0, 0, 0,
    0, 1, 1,
    0, 1, 0,

    // Right
    1, 0, 0,
    1, 1, 0,
    1, 1, 1,

    1, 0, 0,
    1, 1, 1,
    1, 0, 1
};

GLfloat hseg_box_colors[] = {
    // Back
    0, 0, 0,
    0, 1, 0,
    1, 1, 0,

    0, 0, 0,
    1, 1, 0,
    1, 0, 0,

    // Front
    0, 0, 1,
    1, 0, 1,
    1, 1, 1,

    0, 0, 1,
    1, 1, 1,
    0, 1, 1,

    // Top
    0, 1, 0,
    0, 1, 1,
    1, 1, 1,

    0, 1, 0,
    1, 1, 1,
    1, 1, 0,

    // Bottom
    0, 0, 0,
    1, 0, 0,
    1, 0, 1,

    0, 0, 0,
    1, 0, 1,
    0, 0, 1,

    // Left
    0, 0, 0,
    0, 0, 1,
    0, 1, 1,

    0, 0, 0,
    0, 1, 1,
    0, 1, 0,

    // Right
    1, 0, 0,
    1, 1, 0,
    1, 1, 1,

    1, 0, 0,
    1, 1, 1,
    1, 0, 1
};

/**
 * @brief VolumeRenderer::initialize
 *          Initialize all member variables here.
 */
void VolumeRenderer::initialize()
{
    m_window_width  = m_window_height = 0;

    m_repainting_flag = true;
    m_mouse_prevpos = QPoint(0,0);

    for(int i=0;i<16;i++)
        m_camera_matrix[i]=0;

    memset( m_display_box, 0, 6*sizeof(size_t));
    memset( m_dim_scale, 0, 3*sizeof(float));
    memset( m_modelCentroid, 0, 3*sizeof(float));
    memset( m_visTexBounds, 0, 6*sizeof(float));

    for(size_t i=0; i < 3; i++)
        m_voxel_spacing[i] = 1.0;

    m_vbo_box.m_data.clear();
    m_vbo_box.m_data.resize(108);

    m_vbo_colors.m_data.clear();
    m_vbo_colors.m_data.resize(108);

    m_ray_stepsize = 1.0f/1024.0f;

    m_volDim[0] = m_volDim[1] = m_volDim[2] = 0;
    m_grabFrame = false;
    m_grid = nullptr;

    m_vao_cube = 0;
    m_SegmentRenderMode.resize(256); 
    m_SegmentRanges.resize(512); 
    m_VisFlags.resize(32); 
}

/** 
 * @brief VolumeRenderer::VolumeRenderer
 *    
 */
/*
VolumeRenderer::VolumeRenderer()
{
    initialize();
    srand(time(NULL));
}
*/

VolumeRenderer::VolumeRenderer(QWidget *parent)
    : vpCanvas(parent)
{
    initialize();
    //srand(time(NULL));
    srand(10);
}

/**
 * @brief VolumeRenderer::~VolumeRenderer
 *          Clear all resources like shader programs before exiting.
 */
VolumeRenderer::~VolumeRenderer()
{
    makeCurrent();
    glDeleteVertexArrays(1, &m_vao_cube);

    m_vbo_box.Destroy();
    m_vbo_colors.Destroy();

    glDeleteProgram(m_program_cube);
    glDeleteProgram(m_program_ray);

    glDeleteFramebuffers(1, &m_framebuffer);
    glDeleteRenderbuffers(1, &m_renderbuffer);
    glDeleteTextures( 1, &m_backface_buffer);
    glDeleteTextures( 1, &m_final_image);
    //glDeleteTextures( 1, &m_seg_slices.opengl_tex );
    //glDeleteTextures( 1, &m_seg.opengl_tex );

    m_jitter.Destroy();
}

/**
 * @brief VolumeRenderer::Initialize
 *          Initialize the resources required by the volume renderer.
 *          Resources include shaders and glPrograms required by the
 *          technique. Call this function only once before using the
 *          class object and after setting any other input attributes.
 * @return  Returns 0 if failed; 1 if success.
 */
void VolumeRenderer::initializeGL()
{
    glGetError(); // flush out initial opengl error codes

    makeCurrent();
    vpCanvas::initializeGL();

    glGetError(); // flush out initial opengl error codes
    MacroglErrorCheck();
    m_vbo_box.m_data.clear();
    m_vbo_box.m_data.resize(108);

    m_vbo_colors.m_data.clear();
    m_vbo_colors.m_data.resize(108);

    // Updates m_vbo_box and m_vbo_colors data arrays:
    compute_box_vertices();

    MacroglErrorCheck();

    m_program_cube = vpCanvas::load_glsl_program( 
                        m_path_cube_program[0].toLatin1().constData(), 
                        m_path_cube_program[1].toLatin1().constData() );

    m_program_ray = vpCanvas::load_glsl_program(
                        m_path_ray_program[0].toLatin1().constData(), 
                        m_path_ray_program[1].toLatin1().constData() );

    MacroglErrorCheck();

    m_jitter.Destroy();

    if( m_vao_cube )
        glDeleteVertexArrays(1, &m_vao_cube);

    glGenVertexArrays( 1, &m_vao_cube );
    glBindVertexArray( m_vao_cube );

    m_vbo_box.Generate();
    m_vbo_box.TransferToGPU();

    m_vbo_colors.Generate();
    m_vbo_colors.TransferToGPU();

    //m_attrib_backverts.SetVBO(   m_vbo_box.GetOpenGLHandle(), GL_FLOAT, GL_ARRAY_BUFFER );
    //m_attrib_frontverts.SetVBO(  m_vbo_box.GetOpenGLHandle(), GL_FLOAT, GL_ARRAY_BUFFER );
    //m_attrib_backcolors.SetVBO(  m_vbo_colors.GetOpenGLHandle(), GL_FLOAT, GL_ARRAY_BUFFER );
    //m_attrib_frontcolors.SetVBO( m_vbo_colors.GetOpenGLHandle(), GL_FLOAT, GL_ARRAY_BUFFER );

    // cube program attributes binding
    m_attrib_backverts.GLSLBindAttribute( m_program_cube, "vertexPosition" );
    m_attrib_backcolors.GLSLBindAttribute( m_program_cube, "vertexColor" );
    m_uniform_camera_matrix = bind_uniform( m_program_cube, "camera_transform");

    // raycasting program attributes binding
    m_attrib_frontverts.GLSLBindAttribute( m_program_ray, "vertexPosition" );
    m_attrib_frontcolors.GLSLBindAttribute( m_program_ray, "vertexColor" );

//    m_attrib_ray_vertexPosition = bind_attribute( m_program_ray, "vertexPosition");
    m_uniform_ray_model_matrix  = bind_uniform( m_program_ray, "model_transform");
    m_uniform_ray_view_matrix   = bind_uniform( m_program_ray, "view_transform");
    m_uniform_ray_camera_matrix = bind_uniform( m_program_ray, "camera_transform");
    m_uniform_ray_back_tex      = bind_uniform( m_program_ray, "back_texture");
    m_volume.glsl_tex           = bind_uniform( m_program_ray, "volumeTexture");
    m_optical.glsl_tex          = bind_uniform( m_program_ray, "opticalTexture");
    //m_volume_slices.glsl_tex    = bind_uniform( m_program_ray, "volume_texture_array");
    m_jitter.glsl_tex           = bind_uniform( m_program_ray, "jitter_texture");
    m_seg_slices.glsl_tex       = bind_uniform( m_program_ray, "segmentTextureArray");
    m_seg.glsl_tex              = bind_uniform( m_program_ray, "segmentTexture");
    m_uniform_ray_stepsize      = bind_uniform( m_program_ray, "stepSize");
    m_uniform_visTexBounds      = bind_uniform( m_program_ray, "visTexBounds");
    m_uniform_renderModes       = bind_uniform( m_program_ray, "renderModes");
    m_uniform_segmentRanges     = bind_uniform( m_program_ray, "segmentRanges");
    m_uniform_visFlags          = bind_uniform( m_program_ray, "visFlags");
    m_uniform_ray_maxvalue      = bind_uniform( m_program_ray, "maxValue");
    //m_uniform_ray_volDim        = bind_uniform( m_program_ray, "volDim");

    // Initialize the frame buffer object
    RecordWindowSize( width(), height() );

    if( m_backface_buffer )
        glDeleteTextures(1, &m_backface_buffer);

    if( m_final_image )
        glDeleteTextures(1, &m_final_image);

    if( m_framebuffer )
        glDeleteFramebuffers( 1, &m_framebuffer );

    if( m_renderbuffer )
        glDeleteRenderbuffers( 1, &m_renderbuffer );


    init_frame_buffers();

    m_frame_buffs_initialized = true;

    compute_jitter_texture();

    // Compute initial camera matrix
    initialize_camera_position();

    calculate_camera_matrix();
}

void VolumeRenderer::initialize_camera_position()
{
    // Model Matrix

    float norm_dim[3];
    glm::vec3 disp;
    if( GetNormalizedDisplayBox( norm_dim ) )
        disp = glm::vec3(norm_dim[0], norm_dim[1], norm_dim[2]);
    else
        disp = glm::vec3(1.0f,1.0f,1.0f);

    disp = disp*((-0.5f)*float(1.0f));

    //model = glm::rotate( model, glm::radians(-90.0f), glm::vec3(1,0,0) );
    glm::mat4 model(1.0f);
    model = glm::mat4_cast( m_camera.rot_quat );
    model = glm::rotate( model, glm::radians(-90.0f), glm::vec3(1,0,0) ); //Tooth
    //model = glm::rotate( model, glm::radians(+90.0f), glm::vec3(1,0,0) );
    model = glm::rotate( model, glm::radians(+180.0f), glm::vec3(0,0,1) );

    model = glm::translate( model, disp );
    SetModelMatrix( glm::value_ptr(model) );

    // View Matrix
    double* pos   = (double*)m_camera.position;
    double* focus = (double*)m_camera.focus;
    double* up    = (double*)m_camera.up;

    glm::mat4 view  = glm::lookAt( glm::vec3(pos[0],pos[1],pos[2]),
                                   glm::vec3(focus[0],focus[1],focus[2]),
                                   glm::vec3(up[0],up[1],up[2]) );

    SetViewMatrix ( glm::value_ptr(view ) );
}

void VolumeRenderer::calculate_camera_matrix()
{
    glm::mat4 m = glm::make_mat4( GetModelMatrix() );
    glm::mat4 v = glm::make_mat4( GetViewMatrix() );
    glm::mat4 p = glm::make_mat4( GetProjMatrix() );
    glm::mat4 mvp = p * v * m;

    set_matrix( glm::value_ptr(mvp), (float*)m_camera_matrix );
}

/**
 * @brief VolumeRenderer::Display
 * @return
 */
void VolumeRenderer::draw_gl()
{
    if( !m_repainting_flag )
        return;

    MacroglErrorCheck();
    // std::cout << "VolRen draw_gl." << std::endl;

    glBindFramebuffer ( GL_FRAMEBUFFER,  m_framebuffer  );
    MacroglErrorCheck();
    glBindRenderbuffer( GL_RENDERBUFFER, m_renderbuffer );
    MacroglErrorCheck();

    render_backface();

    raycasting_pass();

    //glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //QOpenGLFramebufferObject::bindDefault();
    glBindFramebuffer(GL_FRAMEBUFFER, this->defaultFramebufferObject());

    if( !m_grabFrame )
        render_buffer_to_screen();

    over_draw();
}

void VolumeRenderer::over_draw()
{
}

void VolumeRenderer::resizeGL(int w, int h)
{
    if( !m_repainting_flag )
        return;

    vpCanvas::resizeGL( w, h );

    //makeCurrent();
    if( h==0 )	h = 1;
    if( w==0 )	w = 1;

    RecordWindowSize( w, h);

    if( m_frame_buffs_initialized )
    {
        glDeleteTextures(1, &m_backface_buffer);
        glDeleteTextures(1, &m_final_image);

        glDeleteFramebuffers( 1, &m_framebuffer );
        glDeleteRenderbuffers( 1, &m_renderbuffer );

        init_frame_buffers();
        compute_jitter_texture();
        calculate_camera_matrix();
    }
}

void VolumeRenderer::RecordWindowSize( int width, int height )
{
    width = width == 0? 1: width;
    height = height == 0? 1: height;
    m_window_width  = width;
    m_window_height = height;
}

void VolumeRenderer::init_frame_buffers()
{
    MacroglErrorCheck();

    // Create the to FBO's one for the backside of the volumecube and one for the finalimage rendering
    glGenFramebuffers(1, &m_framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER,m_framebuffer);

    MacroglErrorCheck();

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &m_backface_buffer);
    glBindTexture(GL_TEXTURE_2D, m_backface_buffer);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_window_width, m_window_height, 0, GL_RGB, GL_FLOAT, NULL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_window_width, m_window_height, 0, GL_RGB, GL_FLOAT, NULL);
    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_backface_buffer, 0 );

    MacroglErrorCheck();

    glGenTextures(1, &m_final_image);
    glBindTexture(GL_TEXTURE_2D, m_final_image);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_window_width, m_window_height, 0, GL_RGB, GL_FLOAT, NULL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_window_width, m_window_height, 0, GL_RGB, GL_FLOAT, NULL);

    MacroglErrorCheck();

    glGenRenderbuffers(1, &m_renderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, m_renderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_window_width, m_window_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_renderbuffer);

    MacroglErrorCheck();

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
        MacroWarning("GL_FRAMEBUFFER check failed.");

    MacroglErrorCheck();

    //glBindFramebuffer(GL_FRAMEBUFFER, 0);
    QOpenGLFramebufferObject::bindDefault();

}

bool hseg_toggle_visuals = true;
void VolumeRenderer::render_buffer_to_screen()
{
    glClearColor(1.0,1.0,1.0,1.0);
//    glClearColor(0.8,0.8,0.8,1.0);
    //glClearColor(0.0,0.0,0.0,1.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glLoadIdentity();

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);

    if(hseg_toggle_visuals)
        glBindTexture(GL_TEXTURE_2D, m_final_image);
    else
        glBindTexture(GL_TEXTURE_2D, m_backface_buffer);

    reshape_ortho( m_window_width, m_window_height);

    draw_fullscreen_quad();

    glDisable(GL_TEXTURE_2D);
}

void VolumeRenderer::draw_fullscreen_quad()
{
    glDisable(GL_DEPTH_TEST);
    glBegin(GL_QUADS);

    glTexCoord2f(0,0);
    glVertex2f(0,0);

    glTexCoord2f(1,0);
    glVertex2f(1,0);

    glTexCoord2f(1, 1);
    glVertex2f(1, 1);

    glTexCoord2f(0, 1);
    glVertex2f(0, 1);

    glEnd();
    glEnable(GL_DEPTH_TEST);

}

void VolumeRenderer::reshape_ortho(int w, int h)
{
    if (h == 0) h = 1;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, w, h);
//    GLfloat ratio = 1.0f;
//    gluPerspective( 60.0f, ratio, 0.01f, 100.0f );
    gluOrtho2D(0, 1, 0, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void VolumeRenderer::render_backface()
{
    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_backface_buffer, 0 );
    MacroglErrorCheck();

    glBindVertexArray( m_vao_cube );
    glClearColor(1.0,1.0,1.0,1.0);
//    glClearColor(0.8,0.8,0.8,1.0);
//    glClearColor(0.0,0.0,0.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);

    glUseProgram( m_program_cube );
    glViewport( 0, 0, m_window_width, m_window_height);

    glUniformMatrix4fv( m_uniform_camera_matrix,
                        1,
                        GL_FALSE,
                        m_camera_matrix );

    m_vbo_box.Enable( m_attrib_backverts.GetGLSLHandle() );
    m_vbo_colors.Enable(m_attrib_backcolors.GetGLSLHandle());

    MacroglErrorCheck();

//  Push each element in buffer_vertices to the vertex shader
    size_t sz = m_vbo_box.m_data.size() / 3;
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)sz);

    MacroglErrorCheck();

    m_vbo_box.Disable();
    m_vbo_colors.Disable();

    MacroglErrorCheck();

    glUseProgram( 0 );
    glDisable(GL_CULL_FACE);
}

void VolumeRenderer::raycasting_pass()
{
    MacroglErrorCheck();
    // OpenGL: Start using raycasting shader program
    glUseProgram( m_program_ray);

    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_final_image, 0 );
    MacroglErrorCheck();

//    glEnable(GL_SMOOTH);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glBindVertexArray( m_vao_cube );
    MacroglErrorCheck();

    glUniform1i( m_uniform_ray_maxvalue, m_voltex_scaleoffset[0]);
    //glUniform1iv( m_uniform_ray_maxvalue, 2, m_voltex_scaleoffset);
    //MacroglErrorCheck();
    //glUniform1uiv( m_uniform_ray_volDim, 3, (unsigned int*)m_volDim );
    //int visflags[5]={0,0,0,0,0};
    //m_vstate->GetVisibilityFlags( visflags );
    //glUniform1iv( m_uniform_ray_visible_flags, 5, visflags );

    glUniform1f( m_uniform_ray_stepsize, m_ray_stepsize);
    glUniform1fv( m_uniform_visTexBounds, 6, m_visTexBounds);
    glUniform1uiv( m_uniform_renderModes,   (GLsizei)m_SegmentRenderMode.size(), &m_SegmentRenderMode[0] );
    glUniform1uiv( m_uniform_segmentRanges, (GLsizei)m_SegmentRanges.size(),     &m_SegmentRanges[0] );
    glUniform1uiv( m_uniform_visFlags, (GLsizei)m_VisFlags.size(),     &m_VisFlags[0] );

    glUniform1i( m_volume.glsl_tex,      0 );
    glUniform1i( m_uniform_ray_back_tex, 1 );
    glUniform1i( m_jitter.glsl_tex,      2 );
    glUniform1i( m_optical.glsl_tex,     3 );
    glUniform1i( m_seg_slices.glsl_tex,  4 );
    glUniform1i( m_seg.glsl_tex,  5 );
    //glUniform1i( m_volume_slices.glsl_tex, 4 );

#ifdef COMPUTED_GRADIENTS
    glUniform1i( m_gradients.glsl_tex, 11 );
#endif

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, m_volume.opengl_tex );
    MacroglErrorCheck();

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_backface_buffer);
    MacroglErrorCheck();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_jitter.opengl_tex );
    MacroglErrorCheck();

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_1D_ARRAY, m_optical.opengl_tex );
    MacroglErrorCheck();

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D_ARRAY, m_seg_slices.opengl_tex );
    MacroglErrorCheck();

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_3D, m_seg.opengl_tex );
    MacroglErrorCheck();

    //glBindTexture(GL_TEXTURE_2D_ARRAY, m_volume_slices.opengl_tex );
    //glActiveTexture(GL_TEXTURE5);
    //glBindTexture(GL_TEXTURE_3D, m_seg.opengl_tex );
    //MacroglErrorCheck();


#ifdef COMPUTED_GRADIENTS
    glActiveTexture(GL_TEXTURE11);
    glBindTexture(GL_TEXTURE_3D, m_gradients.opengl_tex );
#endif

    glViewport( 0, 0, m_window_width, m_window_height );

    glClearColor(1.0,1.0,1.0,1.0);
//     glClearColor(0.8,0.8,0.8,1.0);
//    glClearColor(0.0,0.0,0.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    MacroglErrorCheck();

    glUniformMatrix4fv( m_uniform_ray_camera_matrix,
                        1,
                        GL_FALSE,
                        m_camera_matrix );

    MacroglErrorCheck();

    glUniformMatrix4fv( m_uniform_ray_model_matrix,
                        1,
                        GL_FALSE,
                        (const GLfloat*)GetModelMatrix() );

    MacroglErrorCheck();

    glUniformMatrix4fv( m_uniform_ray_view_matrix,
                        1,
                        GL_FALSE,
                        GetViewMatrix() );

    MacroglErrorCheck();

    m_vbo_box.Enable( m_attrib_frontverts.GetGLSLHandle() );
    m_vbo_colors.Enable(m_attrib_frontcolors.GetGLSLHandle());

//  Push each element in buffer_vertices to the vertex shader
    size_t sz = m_vbo_box.m_data.size() / 3;
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)sz);
    MacroglErrorCheck();

    m_vbo_box.Disable();
    m_vbo_colors.Disable();

    // OpenGL: reset to default shader program
    glUseProgram(0);
    glDisable(GL_CULL_FACE);
}


GLuint VolumeRenderer::create_tftexture( size_t width, vpn::tfType* tfunc )
{
    MacroglErrorCheck();

    GLuint tftexture = 0;
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &tftexture );
    glBindTexture( GL_TEXTURE_1D, tftexture );
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );

#ifdef USE_UCHAR_TF
    glTexImage1D( GL_TEXTURE_1D, 0, GL_RGBA8UI, width, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, tfunc );
#else
    glTexImage1D( GL_TEXTURE_1D, 0, GL_RGBA32F, (GLsizei)width, 0, GL_RGBA, GL_FLOAT, tfunc );
#endif

    MacroglErrorCheck();

    return tftexture;
}


/*
void VolumeRenderer::UpdateTFTexture( GLuint tftexture, size_t width, vpn::tfType* tfunc )
{
    if( tftexture )
    {
        glBindTexture( GL_TEXTURE_1D, tftexture );

    #ifdef USE_UCHAR_TF
        glTexImage1D( GL_TEXTURE_1D, 0, GL_RGBA8UI, width, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, tfunc );
    #else
        glTexImage1D( GL_TEXTURE_1D, 0, GL_RGBA32F, width, 0, GL_RGBA, GL_FLOAT, tfunc );
    #endif

        MacroglErrorCheck();
    }
    else
    {
        MacroWarning("Transfer Function Texture not generated.");
    }
}
*/


void VolumeRenderer::SetRayCastingStepSize( float input_value )
{
    m_ray_stepsize = input_value;
    std::cout << "VolumeRenderer:: newStepSize is = " << m_ray_stepsize << std::endl;
}


void VolumeRenderer::SetVoxelSpacing( const double spacing[3] )
{
    vpn::copyvec<double>( spacing, m_voxel_spacing, 3);
}


void VolumeRenderer::SetDisplayBoundingBox( size_t dim[6] )
{
    vpn::copyvec<size_t>( dim, m_display_box, 6 );

    for( int i=0; i < 3; i++)
    {
        // keep min value first
        if( m_display_box[i] > m_display_box[i+3] )
            std::swap(m_display_box[i], m_display_box[i+3]);
    }

    // Update the visible Texture Bounds for shader:
    for(int i=0; i < 3; i++)
    {
        m_visTexBounds[i]   = float(m_display_box[i])   / (float(m_grid->Dim()[i]) - 1.0f);
        m_visTexBounds[i+3] = float(m_display_box[i+3]) / (float(m_grid->Dim()[i]) - 1.0f);
    }

    uint dispDim[3];
    GetDisplayDimensions(dispDim);

    float maxdim = (float)vpn::MaxOfThree(dispDim);
    float newStepSize = 1.0f/(1.0f*maxdim);
    SetRayCastingStepSize( newStepSize );
    compute_box_vertices();

    // Compute initial camera matrix
    initialize_camera_position();

    calculate_camera_matrix();
}

void VolumeRenderer::GetDisplayDimensions( uint dim[3] ) const
{
    NullCheckVoid(dim);
    dim[0] = (uint) (m_display_box[3]-m_display_box[0]);
    dim[1] = (uint) (m_display_box[4]-m_display_box[1]);
    dim[2] = (uint) (m_display_box[5]-m_display_box[2]);
}

/*
void VolumeRenderer::SetTransferFunction(vpn::tfType *tfunc)
{
    if( m_tf[0].opengl_tex )
        UpdateTFTexture( m_tf[0].opengl_tex, m_voltex_scaleoffset, tfunc );
    else
        m_tf[0].opengl_tex = create_tftexture(m_voltex_scaleoffset, tfunc );
}

void VolumeRenderer::SetPancreasTransferFunction(vpn::tfType *tfunc)
{
    if( m_tf[1].opengl_tex )
        UpdateTFTexture( m_tf[1].opengl_tex, m_voltex_scaleoffset, tfunc );
    else
        m_tf[1].opengl_tex = create_tftexture(m_voltex_scaleoffset, tfunc );
}

void VolumeRenderer::SetCystTransferFunction(vpn::tfType *tfunc)
{
    if( m_tf[2].opengl_tex )
        UpdateTFTexture( m_tf[2].opengl_tex, m_voltex_scaleoffset, tfunc );
    else
        m_tf[2].opengl_tex = create_tftexture(m_voltex_scaleoffset, tfunc );
}

void VolumeRenderer::SetDuctTransferFunction(vpn::tfType *tfunc)
{
    m_tf[3].opengl_tex = create_tftexture(m_voltex_scaleoffset, tfunc );
}
*/

/// Set the maximum scalar value in the current volume dataset.
/// input_value: Max scalar value of current volume dataset.
void VolumeRenderer::SetVolumeTextureScaleOffset( int scale, int offset )
{
    m_voltex_scaleoffset[0] = scale;
    m_voltex_scaleoffset[1] = offset;
}

/// Get value of the maximum dimension of the current volume dataset.
/// Return value is the value of the max dimension.
size_t VolumeRenderer::get_max_dimension() const
{
    return vpn::MaxOfThree( m_grid->x(), m_grid->y(), m_grid->z() );
}

///  Get the normalized dimensions of the data volume.
///  Returns 0 if dimensions are not set, return 1 for success.
int VolumeRenderer::GetNormalizedDimensions(float norm_dim[3]) const
{
    float denom = (float)get_max_dimension();
    if( denom > 0 )
    {
        norm_dim[0] = (float)m_grid->x() / denom;
        norm_dim[1] = (float)m_grid->y() / denom;
        norm_dim[2] = (float)m_grid->z() / denom;
        return 1;
    }
    else
    {
        MacroWarning( "Volume data dimensions not set." );
        return 0;
    }
}

void VolumeRenderer::GetModelBox(float box[3])
{
    if( m_modelCentroidCalc )
    {
        box[0] = m_modelCentroid[0];
        box[1] = m_modelCentroid[1];
        box[2] = m_modelCentroid[2];
    }
    else
    {
        float mind[3]={1000,1000,1000};
        float maxd[3]={-1000,-1000,-1000};
        for( size_t i=0; i < m_vbo_box.m_data.size(); i+=3)
        {
            for( int k=0; k < 3; ++k)
            {
                float val = m_vbo_box.m_data[i+k];
                mind[k] = val < mind[k]? val : mind[k];
                maxd[k] = val > maxd[k]? val : maxd[k];
            }
        }

        for( int i=0; i < 3; ++i)
        {
            box[i] = 0.5f*(maxd[i] + mind[i]);
            m_modelCentroid[i] = box[i];
        }
        m_modelCentroidCalc = true;
    }
}

int VolumeRenderer::GetNormalizedDisplayBox(float norm_box[3]) const
{
    float delx = float(m_display_box[3]-m_display_box[0]) * m_voxel_spacing[0];
    float dely = float(m_display_box[4]-m_display_box[1]) * m_voxel_spacing[1];
    float delz = float(m_display_box[5]-m_display_box[2]) * m_voxel_spacing[2];

    float denom = std::max( delx, std::max(dely,delz) );

    if( denom > 0 )
    {
        norm_box[0] = delx / denom;
        norm_box[1] = dely / denom;
        norm_box[2] = delz / denom;
        return 1;
    }
    else
    {
        //MacroWarning( "Invalid display bounding box." );
        return 0;
    }
}

/**
 * @brief VolumeRenderer::compute_box_vertices
 *          Compute the vertices of the box used for ray computations.
 * @return  0 - failure, 1 - success.
 */
int VolumeRenderer::compute_box_vertices()
{
    if( !m_grid )
        return 0;

    if( m_grid->x() == 0 || m_grid->y() == 0 || m_grid->z() == 0 )
    {
        MacroWarning("Dimensions of data not set.");
        return 0;
    }

    int retvalue = 0;
    float norm_dim[3];
    if( GetNormalizedDisplayBox(norm_dim) )
    {
        for( size_t i=0; i < sizeof(hseg_box_colors)/sizeof(GLfloat); i += 3 )
        {
            //vertices
            m_vbo_box.m_data[i+0] = hseg_box_verts[i+0]*norm_dim[0];
            m_vbo_box.m_data[i+1] = hseg_box_verts[i+1]*norm_dim[1];
            m_vbo_box.m_data[i+2] = hseg_box_verts[i+2]*norm_dim[2];
            // colors
            m_vbo_colors.m_data[i+0] = hseg_box_colors[i+0];
            m_vbo_colors.m_data[i+1] = hseg_box_colors[i+1];
            m_vbo_colors.m_data[i+2] = hseg_box_colors[i+2];
        }

        m_modelCentroidCalc = false; // flag to recalculate model box
        retvalue = 1;

        makeCurrent();
        m_vbo_box.TransferToGPU();
        m_vbo_colors.TransferToGPU();

        /*
        std::cout << "Recalculated Box = " << std::endl;
        for(int i=0;i<3;++i)
            std::cout << m_vbo_box.m_data[i] << ", " << m_vbo_box.m_data[i+1]
                    << ", " << m_vbo_box.m_data[i+2] << std::endl;
                    */

    }
    else
    {
        MacroWarning( "Cannot compute rendering box." );
        retvalue = 0;
    }

    return retvalue;
}

void VolumeRenderer::compute_jitter_texture()
{
    if(m_window_width <= 0 || m_window_height <= 0)
    {
        MacroWarning("Cannot create jitter texture. Invalid window dimensions.");
        return;
    }
    int sz = m_window_width * m_window_height;
    GLfloat* data = new GLfloat[sz];
    //int scale = 1000000000;
    //int scale = RAND_MAX;
    int scale = 256;
    for( int i=0; i < sz; i++ )
    {
        int discrete = rand() % scale;
        data[i] = float(discrete) / float(scale);
    }

    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures(1, &m_jitter.opengl_tex);
    glBindTexture(GL_TEXTURE_2D, m_jitter.opengl_tex);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_window_width, m_window_height,
                 0, GL_RED, GL_FLOAT, data);

    MacroglErrorCheck();

    MacroDeleteArray( data );
}

void VolumeRenderer::mouseReleaseEvent(QMouseEvent *event)
{
    NullCheckVoid(event);

    if( event->type() == QWheelEvent::MouseButtonRelease )
    {
        m_mouse_prevpos = QPoint(0,0);
    }

    event->accept();
}

void VolumeRenderer::mouseMoveEvent(QMouseEvent* event)
{
    NullCheckVoid(event);
    bool need_repaint = false;

    if( event->type() == QWheelEvent::MouseMove )
    {
        // If called first time, record current position and return:
        if( m_mouse_prevpos.isNull() )
            m_mouse_prevpos = event->pos();
        // else compute displacement and process color window:
        else
        {
            QPoint new_pos = event->pos();
            QPoint delta = new_pos - m_mouse_prevpos;
            Qt::MouseButtons pressedButtons = event->buttons();

            // left mouse button drag: Rotation
            if( pressedButtons.testFlag(Qt::LeftButton) )
            {
                float change_factor = 0.5f;
                float rot_dy = delta.x() * change_factor;
                float rot_dx = delta.y() * change_factor;

                glm::mat4 y_rot = glm::rotate( glm::mat4(1.0f), glm::radians(rot_dy), glm::vec3(0,1,0) );
                glm::mat4 x_rot = glm::rotate( glm::mat4(1.0f), glm::radians(rot_dx), glm::vec3(1,0,0) );

                glm::quat quat_yrot = glm::quat_cast(y_rot);
                glm::quat quat_xrot = glm::quat_cast(x_rot);

                m_camera.rot_quat = quat_yrot * m_camera.rot_quat;
                m_camera.rot_quat = quat_xrot * m_camera.rot_quat;

                // update the camera and model matrices:
                initialize_camera_position();
                calculate_camera_matrix();

                need_repaint = true;
            }
            // Right mouse button drag: zoom in-out
            else if( pressedButtons.testFlag(Qt::RightButton) )
            {
                float change_factor = 0.01f;
                float disp = delta.y() * change_factor;
                m_camera.position[2] += disp;
                this->clamp( m_camera.position[2], 0.1, 10.0 );

                // update the camera and model matrices:
                initialize_camera_position();
                calculate_camera_matrix();

                need_repaint = true;
            }
            // Middle mouse button drag: panning
            else if( pressedButtons.testFlag(Qt::MiddleButton) )
            {
                float ratio = float(width()) / float(height());
                float x_disp = -ratio*2.0f*float(delta.x()) / float(width());
                float y_disp = 2.0f*float(delta.y()) / float(height());
                //this->clamp( m_displace[0], -1.0f, 1.0f );

                m_camera.position[0] += x_disp;
                m_camera.position[1] += y_disp;
                m_camera.focus[0] += x_disp;
                m_camera.focus[1] += y_disp;

                // update the camera and model matrices:
                initialize_camera_position();
                calculate_camera_matrix();

                need_repaint = true;
            }

            m_mouse_prevpos = new_pos;
        }
    }

    if(need_repaint)
        update();

    event->accept();
}

void VolumeRenderer::SetVolumeDimensions( size_t dim[3] )
{
    for(int i = 0; i <3; ++i)
        m_volDim[i] = (unsigned int)dim[i];
}

void VolumeRenderer::SetGrid(const sjDS::Grid* grid)
{
    if( grid != nullptr )
    {
        m_grid = grid;
        this->SetVoxelSpacing( m_grid->Spacing() );
    }
}


/*
void VolumeRenderer::position_physical_to_opengl(float* phys, float* glpos) const
{
    NullCheckVoid(phys);
    NullCheckVoid(glpos);
    size_t dim[3]={0,0,0};
    m_grid->GetDimensions( dim );

    float norm_dim[3];
    GetNormalizedDisplayBox( norm_dim );

    for( int i=0; i < 3; ++i )
    {
        glpos[i] = (float)(phys[i] / (double(dim[i%3]-1) * m_voxel_spacing[i%3]));
        glpos[i] *= norm_dim[i%3];
    }
}

void VolumeRenderer::position_physical_to_opengl(double* phys, double* glpos) const
{
    NullCheckVoid(phys);
    NullCheckVoid(glpos);
    size_t dim[3]={0,0,0};
    m_grid->GetDimensions( dim );

    float norm_dim[3];
    GetNormalizedDisplayBox(norm_dim);

    for( int i=0; i < 3; ++i )
    {
        glpos[i] = (double)(phys[i] / (double(dim[i%3]-1) * m_voxel_spacing[i%3]));
        glpos[i] *= norm_dim[i%3];
    }
}
*/

int VolumeRenderer::GrabFrame(void** data)
{
    m_grabFrame = true;
    repaint();

    size_t numOfPixels = m_window_width*m_window_height;
    if( !numOfPixels )
    {
        MacroWarning("Invalid window size.");
        return 0;
    }

    *data = malloc(sizeof(unsigned int)*m_window_width*m_window_height);
    glGetTexImage( m_final_image, 0, GL_RGBA8, GL_UNSIGNED_INT, *data );

    m_grabFrame = false;
    return 1;
}

void VolumeRenderer::SetNumberOfSegments(size_t numberOfSegments) 
{ 
    m_SegmentRenderMode.clear();
    m_SegmentRanges.clear(); 
    //if( numberOfSegments > 256 )
    //{
    //    MacroWarning("Render mode array size overflow.");
    //    m_SegmentRenderMode.resize(256); 
    //}
    //else
    //{
    //    m_SegmentRenderMode.resize( numberOfSegments ); 
    //    //m_SegmentRanges.resize( numberOfSegments*2 ); 
    //}
    m_SegmentRenderMode.resize(256); 
    m_SegmentRanges.resize(512); 
}

void VolumeRenderer::SetModeToIgnoreScalars()
{
    m_VisFlags[0] = 1;
}

void VolumeRenderer::SetModeToConsiderScalars()
{
    m_VisFlags[0] = 0;
}

void VolumeRenderer::SetRenderMode(uint label, enumRenderMode mode)
{
    MacroAssert( label < m_SegmentRenderMode.size() );
    MacroAssert( mode >= 0 && mode <= 255 );
    m_SegmentRenderMode[label] = (GLuint)mode;
}

void VolumeRenderer::SetSegmentRange(uint label, uint range[2])
{
    MacroAssert( label < m_SegmentRenderMode.size() );
    m_SegmentRanges[label] = (GLuint)range[0];
    m_SegmentRanges[label+256] = (GLuint)range[1];
}

void VolumeRenderer::SetCubeProgramPath(QString vertex_shader, QString fragment_shader)
{
    m_path_cube_program[0] = vertex_shader;
    m_path_cube_program[1] = fragment_shader;
}

void VolumeRenderer::SetRayProgramPath(QString vertex_shader, QString fragment_shader)
{
    m_path_ray_program[0] = vertex_shader;
    m_path_ray_program[1] = fragment_shader;
}

void VolumeRenderer::SetCamera(const VolumeRenderer::Camera& cam)
{
    m_camera = cam;

    // update the camera and model matrices:
    initialize_camera_position();
    calculate_camera_matrix();
}

void VolumeRenderer::GetDisplayBoundingBox(size_t corners[6]) const
{
    std::copy(m_display_box, m_display_box + 6, corners);
}
