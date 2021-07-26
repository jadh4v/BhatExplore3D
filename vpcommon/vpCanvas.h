#ifndef VPCANVAS_H
#define VPCANVAS_H

#include <QObject>
#include <QWidget>
#include <QOpenGLWidget>
#include <QString>
#include "core/macros.h"
#include "vpStructures.h"
#include "vpTypes.h"
#include "vpCommon.h"

class LIBRARY_API vpCanvas : public QOpenGLWidget
{
Q_OBJECT

private:
    GLfloat *m_viewMatrix, *m_modelMatrix, *m_projMatrix;
    GLint   m_viewMatrixUniform, m_modelMatrixUniform, m_projMatrixUniform;
    GLfloat m_znear, m_zfar, m_fovy;
    bool    m_ortho_mode;

    void init();
    void initialize_camera_matrix();
    void recalculate_projection_matrix();
    void paintGL();
    void paintEvent( QPaintEvent* event );

    // Read an input file to a char array
    static QByteArray file_read( const char *filename );

    /// Main opengl draw routine. Draw 3D stuff by overriding this method.
    virtual void draw_gl();
    /// Override to draw text etc over the 3D GL drawing drawn by draw_gl().
    virtual void draw_over(QPainter* painter);

protected:
    /// cannot override paintGL() - override draw_gl or draw_over instead.
    GLint bind_attribute(GLuint program, const char *attrib_name);
    int   bind_matrix_attributes( GLuint program );
    int   transfer_matrices() const;
    bool  valid_matrix_uniforms() const;

    virtual void initializeGL();
    virtual void resizeGL(int w, int h);

    /// Notifies user of opengl errors. Override to modify basic notification.
    virtual void check_gl_error(const char *str);

    /// print GLSL error log
    static void print_log( GLuint object );

    /// create vertex and fragment shaders by reading the program from a file.
    static GLuint create_shader( const char *filename, GLenum type );

    /// Copy 4x4 matrix from source to destination
    static void set_matrix(const float *src, float *dst);

    /// Transform a 3D point to view plane.
    QPoint get_projected_point( float pt[3]) const;
    int vpCanvas::project_point_to_z(const QPoint& p, const float z, float out[3]) const;

public:
    static GLuint load_glsl_program(const char *vertFile, const char *fragFile);
    static GLint bind_uniform(GLuint program, const char *name);
    explicit vpCanvas(QWidget* parent = 0);
    virtual ~vpCanvas();

    void CaptureScreen();

    void SetModelMatrix( float *matrix4 );
    void SetViewMatrix ( float *matrix4 );
    void SetProjMatrix ( float *matrix4 );

    const float* GetModelMatrix() const { return m_modelMatrix; }
    const float* GetViewMatrix () const { return m_viewMatrix;  }
    const float* GetProjMatrix () const { return m_projMatrix;  }

    MacroSetMember(GLfloat, m_znear, _znear )
    MacroSetMember(GLfloat, m_zfar,  _zfar  )
    MacroSetMember(GLfloat, m_fovy,  _fovy  )

    bool GetOrthoMode() const;
    void SetOrthoMode(bool);

    template<typename T>
    static void clamp( T& value, T v_min, T v_max)
    {
        value = std::min( v_max, value );
        value = std::max( v_min, value );
    }
};

#endif // VPCANVAS_H
