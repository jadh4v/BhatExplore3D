#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

// Lib-GLM (openGL Matrix library)
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <QFile>
#include <QFileDialog>
#include <QPainter>
#include <vpCanvas.h>

void vpCanvas::init()
{
    m_modelMatrix = new GLfloat[16];
    m_viewMatrix  = new GLfloat[16];
    m_projMatrix  = new GLfloat[16];

    m_fovy = 60.0f; m_znear = 0.01f; m_zfar = 100.0f;
    m_modelMatrixUniform = m_viewMatrixUniform = m_projMatrixUniform = -1;
    m_ortho_mode = false;

    initialize_camera_matrix();
}

vpCanvas::vpCanvas(QWidget* parent) : QOpenGLWidget(parent)
{
    init();
    setAutoFillBackground(false);
}

vpCanvas::~vpCanvas()
{
    MacroDeleteArray( m_modelMatrix );
    MacroDeleteArray( m_viewMatrix  );
    MacroDeleteArray( m_projMatrix  );
}


void vpCanvas::initialize_camera_matrix()
{
    // View Matrix
    double pos[3]   = {0,0,1};
    double focus[3] = {0,0,0};
    double up[3]    = {0,1,0};

    // Model Matrix
    glm::mat4 model = glm::translate( glm::mat4(1.0f), glm::vec3( -0.5, -0.5, -0.0) );

    glm::mat4 view  = glm::lookAt( glm::vec3(pos[0],pos[1],pos[2]),
                                   glm::vec3(focus[0],focus[1],focus[2]),
                                   glm::vec3(up[0],up[1],up[2]) );

    // set matrices
    SetModelMatrix( glm::value_ptr(model) );
    SetViewMatrix ( glm::value_ptr(view ) );

    recalculate_projection_matrix();
}

void vpCanvas::recalculate_projection_matrix()
{
    // Projection Matrix
    int w = width();
    int h = height();
    w = w == 0? 1 : w;
    h = h == 0? 1 : h;
    GLfloat ratio = float(w) / float(h);
    glm::mat4 proj;
    if( m_ortho_mode )
        proj = glm::ortho( -1.0f*ratio, 1.0f*ratio, -1.0f, 1.0f, -1.0f, 1.0f);
    else
        proj = glm::perspective( glm::radians(m_fovy), ratio, m_znear, m_zfar );

    SetProjMatrix ( glm::value_ptr(proj ) );
}

void vpCanvas::initializeGL()
{
    makeCurrent();
    GLenum err = glewInit();

    if( err != GLEW_OK )
    {
        MacroWarning("Glew init failed.");
        return;
    }
}

void vpCanvas::resizeGL(int w, int h)
{
    makeCurrent();
    if( h==0 )	h = 1;
    if( w==0 )	w = 1;

    GLfloat ratio = float(w)/float(h);
    MacroglErrorCheck();
    glViewport( 0, 0, w, h );
    MacroglErrorCheck();
    recalculate_projection_matrix();
}

void vpCanvas::paintEvent( QPaintEvent* )
{
    makeCurrent();
    MacroglErrorCheck();

    QColor qtBgColor = QColor::fromRgbF(0,0,0,1);
    qtBgColor = qtBgColor.darker();
    glClearColor(qtBgColor.redF(), qtBgColor.greenF(), qtBgColor.blueF(), qtBgColor.alphaF());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    MacroglErrorCheck();

    // Draw the main 3D scene
    draw_gl();
    MacroglErrorCheck();

    MacroglErrorCheck();

    // Paint over the 3D scene
    QPainter painter(this);
    draw_over( &painter );
    MacroglErrorCheck();
    painter.end();
}

void vpCanvas::paintGL()
{
    draw_gl();
}

void vpCanvas::draw_gl()
{
    // empty implementation
}

void vpCanvas::draw_over(QPainter* painter)
{
    Q_UNUSED(painter);
    // empty implementation
}

void vpCanvas::check_gl_error( const char* str )
{
    GLenum err = glGetError();
    if( err == GL_INVALID_ENUM )
    {
        MacroWarning( str << " GL_INVALID_ENUM ");
    }
    else if( err == GL_INVALID_VALUE )
    {
        MacroWarning( str << " GL_INVALID_VALUE ");
    }
    else if( err == GL_INVALID_OPERATION )
    {
        MacroWarning( str << " GL_INVALID_OPERATION ");
    }
}

/**
 * @brief vpRenderer::bind_attribute
 *          bind an attribute name of a shader program to an attribute
 *          handle in the form of a GLint.
 * @param program The GL program to bind with.
 * @param attrib_name Name of the attribute in the shader program
 * @return Returns the attribute handler (GLint) if successful.
 *          Returns -1 if failure.
 */
GLint vpCanvas::bind_attribute( GLuint program,
                                        const char* attrib_name )
{
    GLint attribute = glGetAttribLocation( program, attrib_name );
    if( attribute == -1 )
    {
        MacroWarning( "Could not bind attribute:" << attrib_name );
    }
    return attribute;
}

/**
 * @brief vpRenderer::bind_uniform
 *          Bind a uniform name of a shader program to a uniform
 *          handle in the form of a GLint.
 * @param program   Handle of the GL program to bind with.
 * @param name      Name of the uniform variable in the shader program
 * @return          Returns handle if success, else -1.
 */
GLint vpCanvas::bind_uniform( GLuint program,
                                        const char* name )
{
    GLint uniform_handle = glGetUniformLocation( program, name );
    if( uniform_handle == -1 )
    {
        MacroWarning( "Could not bind uniform:" << name );
    }
    return uniform_handle;
}

int vpCanvas::bind_matrix_attributes( GLuint program )
{
    this->makeCurrent();
    //MacroConfirmOrReturn( program, 0 );
    m_modelMatrixUniform = bind_uniform( program, "modelMatrix" );
    m_viewMatrixUniform  = bind_uniform( program, "viewMatrix" );
    m_projMatrixUniform  = bind_uniform( program, "projMatrix" );

    if( m_modelMatrixUniform == -1 || m_viewMatrixUniform == -1 || m_projMatrixUniform == -1 )
        return 0;

    return 1;
}

bool vpCanvas::valid_matrix_uniforms() const
{
    return ( m_modelMatrixUniform >= 0 && m_viewMatrixUniform >= 0 && m_projMatrixUniform >= 0 );
}

int vpCanvas::transfer_matrices() const
{
    if( valid_matrix_uniforms() )
    {
        glUniformMatrix4fv( m_modelMatrixUniform, 1, GL_FALSE, m_modelMatrix );
        glUniformMatrix4fv( m_viewMatrixUniform,  1, GL_FALSE, m_viewMatrix  );
        glUniformMatrix4fv( m_projMatrixUniform,  1, GL_FALSE, m_projMatrix  );
        return 1;
    }
    else
        return 0;
}

void vpCanvas::set_matrix( const float* src, float* dst )
{
    NullCheckVoid( src );
    NullCheckVoid( dst );
    for( int i=0; i < 16; i++)
        dst[i] = (GLfloat)src[i];
}

void vpCanvas::SetModelMatrix( float *matrix4 )
{
    set_matrix( matrix4, m_modelMatrix );
}

void vpCanvas::SetViewMatrix( float *matrix4 )
{
    set_matrix( matrix4, m_viewMatrix );
}

void vpCanvas::SetProjMatrix( float *matrix4 )
{
    set_matrix( matrix4, m_projMatrix );
}

/// Read a text file into a char buffer. Caller will have to free the memory.
/// Returns a null terminated char buffer containing the file contents.
QByteArray vpCanvas::file_read(const char* filename)
{
    NullCheck( filename, QByteArray() );

    QFile shaderFile(filename);

    shaderFile.open( QIODevice::ReadOnly );

    // Check if file is ready to read.
    if( !shaderFile.isOpen() || shaderFile.atEnd() || !shaderFile.isReadable() )
    {
        MacroWarning("Cannot open shaderFile: " << filename );
        return QByteArray();
    }

    QByteArray array = shaderFile.readAll();
    return array;
}

///
/// Display compilation errors from the OpenGL shader compiler
void vpCanvas::print_log(GLuint object)
{
    GLint log_length = 0;
    if (glIsShader(object))
        glGetShaderiv(object, GL_INFO_LOG_LENGTH, &log_length);
    else if (glIsProgram(object))
        glGetProgramiv(object, GL_INFO_LOG_LENGTH, &log_length);
    else
    {
        fprintf(stderr, "printlog: Not a shader or a program\n");
        return;
    }

    char* log = new char[log_length];

    if (glIsShader(object))
        glGetShaderInfoLog(object, log_length, NULL, log);
    else if (glIsProgram(object))
        glGetProgramInfoLog(object, log_length, NULL, log);

    std::cout << log << std::endl;
    delete[] log;
}

///
/// Compile the shader from file 'filename', with error handling
GLuint vpCanvas::create_shader(const char* filename, GLenum type)
{
    QByteArray fileContents = file_read( filename );
    const GLchar* source = fileContents.data();
    NullCheck( source, 0);

    GLuint res = glCreateShader(type);
    glShaderSource(res, 1, &source, NULL);

    glCompileShader(res);
    GLint compile_ok = GL_FALSE;
    glGetShaderiv(res, GL_COMPILE_STATUS, &compile_ok);
    if (compile_ok == GL_FALSE)
    {
        fprintf(stderr, "%s:", filename);
        print_log(res);
        glDeleteShader(res);
        return 0;
    }

    return res;
}

QPoint vpCanvas::get_projected_point(float pt[3]) const
{
    glm::mat4 m = glm::make_mat4( (float*)m_modelMatrix );
    glm::mat4 v = glm::make_mat4( (float*)m_viewMatrix );
    glm::mat4 p = glm::make_mat4( (float*)m_projMatrix );

    float pt4[4];
    pt4[0] = pt[0];
    pt4[1] = pt[1];
    pt4[2] = pt[2];
    pt4[3] = 1.0f;

    glm::vec4 v4 = glm::make_vec4( pt4 );
    glm::mat4 mvp = p*v*m;
    glm::vec4 t = mvp*v4;

    int w = this->width();
    int h = this->height();

    t[0] = 0.5f*w*(t[0] / t[3] + 1.0f);
//    t[1] = 0.5f*h*(t[1] / t[3] + 1.0f);
    t[1] = 0.5f*h*(2.0f - (t[1] / t[3] + 1.0f));

    QPoint ret( (int)t[0], (int)t[1] );

    return ret;
}

int vpCanvas::project_point_to_z(const QPoint& s, const float sz, float out[3]) const
{
    NullCheck(out, 0);

    int w = QOpenGLWidget::width();
    int h = QOpenGLWidget::height();

    double dx = 2.0 * double(s.x()) / double(w)     - 1.0;
    double dy = 2.0 * double(h - s.y()) / double(h) - 1.0;
    glm::mat4 m = glm::make_mat4( (float*)GetModelMatrix());
    glm::mat4 v = glm::make_mat4( (float*)GetViewMatrix());
    glm::mat4 p = glm::make_mat4( (float*)GetProjMatrix());
    glm::mat4 mvp = p*v*m;

    glm::mat4 inv_mvp = glm::inverse(mvp);
    glm::vec4 o = inv_mvp * glm::vec4(dx, dy, sz, 1.0);

    out[0] = (GLfloat) o.x;
    out[1] = (GLfloat) o.y;
    out[2] = (GLfloat) 0.0f;

    return 1;
}

void vpCanvas::CaptureScreen()
{
    QImage snap = grabFramebuffer();
    QString save_file = QFileDialog::getSaveFileName( this, "Save Snapshot", QDir::currentPath() );
    snap.save( save_file, "*.png", 100 );
}

void vpCanvas::SetOrthoMode(bool)
{
    m_ortho_mode = true;
    recalculate_projection_matrix();
}

bool vpCanvas::GetOrthoMode() const
{
    return m_ortho_mode;
}

GLuint vpCanvas::load_glsl_program( const char* vertFile, const char *fragFile )
{
    GLuint hdl_program=0;
    GLuint hdl_vert_shader = create_shader( vertFile, GL_VERTEX_SHADER );
    GLuint hdl_frag_shader = create_shader( fragFile, GL_FRAGMENT_SHADER );

    NullCheck( hdl_vert_shader, 0 );
    NullCheck( hdl_frag_shader, 0 );

    MacroglErrorCheck();
    hdl_program = glCreateProgram();
    MacroglErrorCheck();
    glAttachShader( hdl_program, hdl_vert_shader );
    MacroglErrorCheck();
    glAttachShader( hdl_program, hdl_frag_shader );
    MacroglErrorCheck();
    glLinkProgram( hdl_program );
    MacroglErrorCheck();

    GLint link_ok = GL_FALSE;
    glGetProgramiv( hdl_program, GL_LINK_STATUS, &link_ok );
    MacroglErrorCheck();
    if( !link_ok )
    {
        MacroWarning("glLinkProgram: ");
        print_log( hdl_program );
        return 0;
    }

    return hdl_program;
}













