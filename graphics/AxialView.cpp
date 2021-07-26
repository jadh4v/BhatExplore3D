#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

// Lib-GLM (openGL Matrix library)
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <QWheelEvent>

#include "AxialView.h"
#include "Core/macros.h"

using std::cout;
using std::endl;

const GLfloat AxialView::cSlice[] = {    0, 0, 0,
                                         1, 0, 0,
                                         0, 1, 0,
                                         1, 0, 0,
                                         1, 1, 0,
                                         0, 1, 0
};

const GLfloat AxialView::cTexCoord[] = { 0, 1, 0,
                                         1, 1, 0,
                                         0, 0, 0,
                                         1, 1, 0,
                                         1, 0, 0,
                                         0, 0, 0
};

void AxialView::initialize()
{
    m_color_window[0] = m_color_window[1] = 0;
    m_voltex_maxvalue = 0;
    m_pick = 0;
    m_segoutline = 0;
    m_current_slice = 0;
    m_grid = NULL;
    m_scaling_factor = 1;
    m_displace[0] = m_displace[1] = 0;
    m_spacing[0] = m_spacing[1] = 1.0;
    m_repainting_flag = true;
    m_scribbling = false;
    m_ScribbleChanged = false;
    m_interpolation_mode = 0;

    m_mouse_prevpos  = QPoint(0,0);
    m_selected_point[0] = m_selected_point[1] = m_selected_point[2] = 0.0f;

    memset( (void*) m_quad_vertices, 0, sizeof(m_quad_vertices) );
    memset( (void*) m_quad_colors, 0, sizeof(m_quad_colors) );

    m_program_slice = 0;
    m_vbo_verts     = 0;
    m_vbo_colors    = 0;

    m_attrib_verts          = -1;
    m_uniform_pick          = -1;
    m_uniform_interpolation = -1;
    m_uniform_segoutline    = -1;
    m_attrib_colors         = -1;
    m_uniform_window        = -1;
    m_uniform_maxvalue      = -1;
    m_uniform_slice_number  = -1;
}

AxialView::~AxialView()
{
}

AxialView::AxialView(QWidget *parent)
    : vpCanvas(parent)
{
    initialize();

    reconstruct_slice_plane();

    SetOrthoMode(true);
}

void AxialView::initializeGL()
{
    makeCurrent();
    vpCanvas::initializeGL();

    glGenVertexArrays( 1, &m_vao );
    glBindVertexArray( m_vao );

    m_program_slice = vpCanvas::load_glsl_program( 
        m_path_to_slice_program[0].toLatin1().constData(), 
        m_path_to_slice_program[1].toLatin1().constData() );

    glGenBuffers( 1, &m_vbo_verts );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo_verts );
    glBufferData( GL_ARRAY_BUFFER,
                      sizeof(m_quad_vertices),
                      m_quad_vertices,
                      GL_STATIC_DRAW );

    glGenBuffers( 1, &m_vbo_colors );
    glBindBuffer( GL_ARRAY_BUFFER, m_vbo_colors );
    glBufferData( GL_ARRAY_BUFFER,
                      sizeof(m_quad_colors),
                      m_quad_colors,
                      GL_STATIC_DRAW );

    MacroglErrorCheck();

    // slice program attributes binding
    m_attrib_verts           = bind_attribute( m_program_slice, "vertexPosition");
    m_attrib_colors          = bind_attribute( m_program_slice, "vertexColors");
    m_volume.glsl_tex        = bind_uniform  ( m_program_slice, "volumeTexture");
    m_segment.glsl_tex       = bind_uniform  ( m_program_slice, "segmentTexture");
    m_segment_array.glsl_tex = bind_uniform  ( m_program_slice, "segmentTextureArray");
    m_gradient.glsl_tex      = bind_uniform  ( m_program_slice, "gradientTexture");
    m_optical.glsl_tex       = bind_uniform  ( m_program_slice, "opticalTexture");
    m_selection.glsl_tex     = bind_uniform  ( m_program_slice, "selectionTexture");
    m_uniform_maxvalue       = bind_uniform  ( m_program_slice, "maxValue");
    m_uniform_window         = bind_uniform  ( m_program_slice, "window");
    m_uniform_pick           = bind_uniform  ( m_program_slice, "pick");
    m_uniform_slice_number   = bind_uniform  ( m_program_slice, "sliceNumber");
    m_uniform_segoutline     = bind_uniform  ( m_program_slice, "segOutlineEnabled");
    m_uniform_interpolation  = bind_uniform  ( m_program_slice, "interpolation");

    bind_matrix_attributes(m_program_slice);
}

void AxialView::SetGrid(const sjDS::Grid * g)
{
    m_grid = g;
    // Allocate memory for selection mask based on the grid-size of raw data.
    m_selectionMask.Resize(*m_grid);
    _CreateSelectionTexture();
}

void AxialView::wheelEvent(QWheelEvent* event)
{
    NullCheckVoid(event);
    bool need_repaint = false;

    if( event->type() == QWheelEvent::Wheel )
    {
        QPoint degrees = event->angleDelta() / 8;
        if( !degrees.isNull() )
        {
            QPoint steps = degrees / 15;
            m_current_slice += steps.y();
            m_current_slice = std::max( 0, m_current_slice );
            need_repaint = true;

            if( !m_grid )
                MacroWarning("Grid is a required input object for AxialView.");
            else
            {
                int z = int(m_grid->z());

                MacroAssert(z > 0);

                if(z != 0) z--;

                m_current_slice = std::min( m_current_slice, z );
                emit(sign_sliceChanged(m_current_slice));
            }
        }
    }

    if(need_repaint)
        update();

    event->accept();
}

void AxialView::_AdjustColorWindow(const QPoint& delta)
{
    // or change color-window settings.
    float change_factor = 4.0f;
    m_color_window[0] -= delta.x() * change_factor;
    m_color_window[1] += delta.y() * change_factor;

    m_color_window[0] = std::max(0.0f, m_color_window[0]);
    m_color_window[1] = std::max(5.0f, m_color_window[1]);

    float upperLimit = 65000.0f;
    m_color_window[0] = std::min(upperLimit, m_color_window[0]);
    m_color_window[1] = std::min(upperLimit, m_color_window[1]);
}

void AxialView::_Zoom(const QPoint& delta)
{
    float change_factor = 0.01f;
    m_scaling_factor += delta.y() * change_factor;
    this->clamp( m_scaling_factor, 0.1f, 20.0f );
    update_model_matrix();
}

void AxialView::_Pan(const QPoint& delta)
{
    float ratio = float(width()) / float(height());
    m_displace[0] += ratio*2.0f*float(delta.x()) / float(width());
    m_displace[1] -= 2.0f*float(delta.y()) / float(height());
    update_model_matrix();
}

void AxialView::mouseMoveEvent(QMouseEvent *event)
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

            // left mouse button: Color-Window settings / or scribbling mode
            if( pressedButtons.testFlag(Qt::LeftButton) )
            {
                // Go into scribbling mode and select clicked voxels,
                if (m_scribbling)
                    _FindClickedVoxel(event);
                else
                    _AdjustColorWindow(delta);

                need_repaint = true;
            }
            // Right Mouse Button: Zoom
            else if( pressedButtons.testFlag(Qt::RightButton) )
            {
                _Zoom(delta);
                need_repaint = true;
            }
            // Middle Mouse Button: Pan
            else if( pressedButtons.testFlag(Qt::MiddleButton) )
            {
                _Pan(delta);
                need_repaint = true;
            }

            m_mouse_prevpos = new_pos;
        }
    }

    // repaint the viewport if update was performed.
    if(need_repaint)
        update();

    event->accept();
}

void AxialView::mouseReleaseEvent(QMouseEvent *event)
{
    NullCheckVoid(event);

    Qt::MouseButtons pressedButtons = event->buttons();
    if( event->type() == QWheelEvent::MouseButtonRelease )
    {
        //if( m_scribbling )
        //    _FindClickedVoxel(event);

        m_mouse_prevpos = QPoint(0,0);
    }

    event->accept();
}

void AxialView::keyPressEvent(QKeyEvent * event)
{
    cout << "AxialView: key was pressed." << endl;
    if(event->key() == Qt::Key_T)
    {
        m_interpolation_mode = m_interpolation_mode == 0? 1 : 0;
        if( m_interpolation_mode == 0 )
            MacroMessage("Interpolation mode: Linear");
        else
            MacroMessage("Interpolation mode: Bi-Cubic");
    }

    QOpenGLWidget::keyPressEvent(event);
}

void AxialView::_FindClickedVoxel(QMouseEvent* event)
{
    NullCheckVoid(event);
    QPoint new_pos = event->pos();
    QPoint delta = new_pos - m_mouse_prevpos;
    bool need_repaint = false;
    if (GetOrthoMode())// && event->button() == Qt::LeftButton)
    {
        project_point_to_z(new_pos, 0.0f, m_selected_point);
        m_selected_point[2] = (GLfloat) 0.0f;

        // Calculate the clicked voxel based on the 3D point and drawing Quad dimensions.
        size_t voxel[3] = { 0,0,0 };    // grid coordinates of a voxel.
        double dx = (double)(m_quad_vertices[12] - m_quad_vertices[0]);  // x-size of quad
        double dy = (double)(m_quad_vertices[13] - m_quad_vertices[1]);  // y-size of quad

        // calculate normalized position of selected voxel w.r.t. the Quad.
        double nx = double((m_selected_point[0] - m_quad_vertices[0]) / dx);
        double ny = double((m_selected_point[1] - m_quad_vertices[1]) / dy);

        if (nx > 0 && nx < 1 && ny > 0 && ny < 1)
        {
            voxel[0] = (size_t)(nx * double(m_grid->x()));
            voxel[1] = (size_t)(ny * double(m_grid->y()));
            voxel[2] = (size_t)m_current_slice;
            //std::cout << "Selected voxel = " << voxel[0] << "," << voxel[1] << "," << voxel[2] << std::endl;

            // Set the voxel only if it has not been set before (for efficiency).
            if (m_selectionMask.Get(voxel) == 0)
            {
                m_selectionMask.Set(voxel);     // set the bit in main-memory.
                _UpdateSelectionTexture(true);      // transfer data to opengl texture.
                m_ScribbleChanged = true;
            }
        }
        //else
        //    MacroWarning("Clicked outside the quad.");

        // repaint the viewport if update was performed.
        if (need_repaint)
            update();
    }
    else
        MacroWarning("Voxel picking has not been tested for perspective projection.");
}


void AxialView::draw_gl()
{
    if( m_volume.opengl_tex == 0 || !m_repainting_flag )
    {
        glClearColor( 0.3f, 0.3f, 0.3f, 1.0f ); // set background color
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    }
    else
    {
        //std::cout << "AxialView draw_gl." << std::endl;
        // Use default shader
        MacroglErrorCheck();
        glBindVertexArray( m_vao );
        glUseProgram(m_program_slice);

        MacroglErrorCheck();
        glViewport( 0, 0, width(), height() );

        MacroglErrorCheck();
        glClearColor( 0.3f, 0.3f, 0.3f, 1.0f ); // set background color
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        MacroglErrorCheck();
        vpCanvas::transfer_matrices();

        glUniform1i( m_uniform_maxvalue, m_voltex_maxvalue);
        glUniform1i( m_uniform_pick, m_pick );
        glUniform1i( m_uniform_slice_number, m_current_slice );
        glUniform1i( m_uniform_segoutline, m_segoutline );
        glUniform1i( m_uniform_interpolation, m_interpolation_mode );
        glUniform1fv( m_uniform_window, 2, m_color_window );
        MacroglErrorCheck();

        // Texture for volumetric image.
        glUniform1i( m_volume.glsl_tex, 0 );
        MacroglErrorCheck();
        glActiveTexture(GL_TEXTURE0);
        MacroglErrorCheck();
        //glBindTexture(GL_TEXTURE_3D, m_volume.opengl_tex );
        glBindTexture(GL_TEXTURE_2D_ARRAY, m_volume.opengl_tex );
        MacroglErrorCheck();

        // Texture for volumetric image.
        glUniform1i( m_selection.glsl_tex, 1 );
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, m_selection.opengl_tex );
        MacroglErrorCheck();

        // Texture for gradients of image (f_x, f_y, f_xy).
        glUniform1i( m_gradient.glsl_tex, 2 );
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D_ARRAY, m_gradient.opengl_tex );
        MacroglErrorCheck();

        if( m_segoutline )
        {
            // Texture for segmentation labels.
            glUniform1i( m_segment_array.glsl_tex, 3 );
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D_ARRAY, m_segment_array.opengl_tex );
        MacroglErrorCheck();

            // Texture for optical properties of each label.
            glUniform1i( m_optical.glsl_tex, 4 );
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_1D_ARRAY, m_optical.opengl_tex );
        MacroglErrorCheck();

            // Texture for segmentation labels.
            glUniform1i( m_segment.glsl_tex, 5 );
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_3D, m_segment.opengl_tex );
        MacroglErrorCheck();
        }
        MacroglErrorCheck();


        // Bind vertices of slice plane
        //GLuint vert_attrib = 0, color_attrib = 2;
        glEnableVertexAttribArray( m_attrib_verts );
        MacroglErrorCheck();
        glBindBuffer( GL_ARRAY_BUFFER, m_vbo_verts );
        MacroglErrorCheck();
        glVertexAttribPointer( m_attrib_verts, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<const GLvoid*>(0) );
        MacroglErrorCheck();

        // Bind color buffer of slice plane
        glEnableVertexAttribArray( m_attrib_colors );
        glBindBuffer( GL_ARRAY_BUFFER, m_vbo_colors );
        glVertexAttribPointer( m_attrib_colors, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<const GLvoid*>(0) );
        MacroglErrorCheck();

        // Push each element in buffer_vertices to the vertex shader
        size_t sz = sizeof(m_quad_vertices) /(3*sizeof(GLfloat));
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)sz);
        MacroglErrorCheck();

        glDisableVertexAttribArray(m_attrib_verts);
        glDisableVertexAttribArray(m_attrib_colors);

        // Reset to default shader
        glUseProgram( 0 );
        glBindBuffer( GL_ARRAY_BUFFER, 0 );

        /*
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_LINE_SMOOTH);
        glDisable(GL_DEPTH_TEST);
        glColor3f( 1.0f, 1.0f, 1.0f );
        glPointSize(5.0f);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glm::mat4 m = glm::make_mat4( (float*)GetModelMatrix());
        glm::mat4 v = glm::make_mat4( (float*)GetViewMatrix());
        glm::mat4 p = glm::make_mat4( (float*)GetProjMatrix());

        float pt[4];
        pt[0] = m_selected_point[0];
        pt[1] = m_selected_point[1];
        pt[2] = m_selected_point[2];
        pt[3] = 1.0f;
        glm::mat4 mvp = p*v*m;
        glm::vec4 glm_p = mvp * glm::make_vec4(pt);
        GLfloat drawPoint[3] = { glm_p.x, glm_p.y, glm_p.z };

        glBegin(GL_POINTS);
        glVertex3fv(drawPoint);
        glEnd();

        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        */

    }
}

void AxialView::draw_over(QPainter * painter)
{
}

void AxialView::reconstruct_slice_plane()
{
    // vertices of the slice plane
    for( size_t i=0; i < 18; i++ )
        m_quad_vertices[i] = cSlice[i];

    // colors of the slice plane
    for( size_t i=0; i < 18; i++ )
        m_quad_colors[i] = cTexCoord[i];

    slice_scaling();
}

void AxialView::SetVolumeTextureArray( uint handle )
{
    m_volume.opengl_tex = handle;
}

void AxialView::SetSegmentTexture( uint handle )
{
    m_segment.opengl_tex = handle;
    m_segoutline = 1;
}

void AxialView::SetSegmentTextureArray( uint handle )
{
    m_segment_array.opengl_tex = handle;
    m_segoutline = 1;
}

void AxialView::SetGradientTextureArray( uint handle )
{
    m_gradient.opengl_tex = handle;
}

void AxialView::SetOpticalTexture( uint handle )
{
    m_optical.opengl_tex = handle;
}

void AxialView::SetDicomColorWindow( float window[2] )
{
    NullCheckVoid(window);

    m_color_window[0] = 1024 + window[0];
    m_color_window[1] = window[1];
}

void AxialView::SetSpacing(const double spacing[2])
{
    m_spacing[0] = spacing[0];
    m_spacing[1] = spacing[1];
    reconstruct_slice_plane();

    makeCurrent();
    MacroglErrorCheck();
    if (m_vbo_verts)
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo_verts);
        glBufferData(GL_ARRAY_BUFFER,
            sizeof(m_quad_vertices),
            m_quad_vertices,
            GL_STATIC_DRAW);

        MacroglErrorCheck();
    }
    else
    {
        MacroWarning("GL context not initialized.");
    }
}

void AxialView::slot_updateSlice(int slice_number)
{
    m_current_slice = slice_number;
    m_current_slice = std::max( 0, m_current_slice );
    int z = int(m_grid->z());
    if(z != 0) z--;
    m_current_slice = std::min( m_current_slice, z );
    update();
}

void AxialView::update_model_matrix()
{
    glm::vec3 scaling_vec(m_scaling_factor, m_scaling_factor, 1);

    glm::mat4 model(1);
    model = glm::scale( model, scaling_vec );
    model = glm::translate( model, glm::vec3( m_displace[0]/m_scaling_factor-0.5, m_displace[1]/m_scaling_factor-0.5, -0.0) );

    this->SetModelMatrix( glm::value_ptr(model) );
}

int AxialView::get_slice_aspect_ratio(double ratio[2]) const
{
    if( !m_grid )
    {
        MacroWarning("No grid object.");
        return 1;
    }

    double width  = (double) m_grid->x() * m_spacing[0];
    double height = (double) m_grid->y() * m_spacing[1];

    if( width == 0 || height == 0)
    {
        MacroWarning("Avoiding divide by zero.");
        return 0;
    }

    ratio[0] = width  / qMax( width, height );
    ratio[1] = height / qMax( width, height );

    if( ratio[0] > ratio[1])
        return 1;
    else
        return 2;
}

void AxialView::slice_scaling()
{
//    if( validate_dimensions() && validate_orientation() )
//            && m_orientation != vpn::SliceOrientationArbitrary )
    {
        NullCheckVoid(m_quad_vertices);

        double aspect[2]={0,0};
        if( get_slice_aspect_ratio( aspect ) == 0 )
            return;

        for(int i=0; i < 6; i++)
        {
            m_quad_vertices[i*3+0] *= aspect[0];
            m_quad_vertices[i*3+1] *= aspect[1];
        }
    }
}

int AxialView::_CreateSelectionTexture()
{
    size_t dim[3] = { 0,0,0 };
    m_selectionMask.GetByteDimensions(dim);
    GLvoid* volumeData = (GLvoid*)m_selectionMask.GetRawPointer();

    makeCurrent();
    m_selection.Destroy();

    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &m_selection.opengl_tex );
    glBindTexture( GL_TEXTURE_3D, m_selection.opengl_tex );
    //glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER );

    glTexImage3D( GL_TEXTURE_3D, 0, GL_R8UI, (GLsizei)dim[0], (GLsizei)dim[1],
                  (GLsizei)dim[2], 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, volumeData );

    GLenum r = MacroglErrorCheck();
    if( r == GL_NO_ERROR )
        return 1;
    else
        return 0;
}

int AxialView::_UpdateSelectionTexture(bool currentSliceOnly )
{
    size_t dim[3] = { 0,0,0 };
    m_selectionMask.GetByteDimensions(dim);
    GLvoid* volumeData = nullptr;
    if( currentSliceOnly )
        volumeData = (GLvoid*)m_selectionMask.GetRawPointerToSlice(m_current_slice);
    else
        volumeData = (GLvoid*)m_selectionMask.GetRawPointer();

    NullCheck(volumeData, 0);

    int ret = 1;
    // if texture handle is valid:
    if( m_selection.opengl_tex ) 
    {
        makeCurrent();
        glBindTexture( GL_TEXTURE_3D, m_selection.opengl_tex  );
        if( currentSliceOnly )
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, m_current_slice, (GLsizei)dim[0], (GLsizei)dim[1], (GLsizei)1, GL_RED_INTEGER, GL_UNSIGNED_BYTE, volumeData);
        else
            glTexImage3D( GL_TEXTURE_3D, 0, GL_R8UI, (GLsizei)dim[0], (GLsizei)dim[1], (GLsizei)dim[2], 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, volumeData );

        GLenum r = MacroglErrorCheck();
        if( r != GL_NO_ERROR )
           ret = 0;
    }
    else
    {
        MacroWarning("SeletionMask texture not defined.");
        ret = 0;
    }

    return ret;
}

void AxialView::slot_EnableScribbling(bool state)
{
    m_scribbling = state;
}

void AxialView::slot_ClearScribbling(bool state)
{
    Q_UNUSED(state);
    m_selectionMask.ClearAll();
    _UpdateSelectionTexture(false);
    update();
    m_ScribbleChanged = true;
}

void AxialView::SetSliceShaderProgram(QString vertex_shader, QString fragment_shader)
{
    m_path_to_slice_program[0] = vertex_shader;
    m_path_to_slice_program[1] = fragment_shader;
}

void AxialView::ToggleInterpolation()
{
    m_interpolation_mode = (m_interpolation_mode+1)%3;
    switch( m_interpolation_mode )
    {
    case 0:  MacroMessage("Interpolation mode: PWC");        break;
    case 1:  MacroMessage("Interpolation mode: Linear");     break;
    case 2:  MacroMessage("Interpolation mode: Bi-Cubic");   break;
    default: MacroMessage("Interpolation mode: Undefined");  break;
    }
}
