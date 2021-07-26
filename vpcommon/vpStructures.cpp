#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

#include <iostream>

#include "vpStructures.h"


unsigned int glErrorCheck( const char* file, const int line )
{
    const  int max_errors = 10;
    GLenum curr_err = 0, prev_err = 0;
    for (int i = 0; i < max_errors; ++i)
    {
        curr_err = glGetError();
        switch (curr_err)
        {
        case GL_NO_ERROR:
            return prev_err;
        case GL_INVALID_ENUM:
            std::cout << file << " Line: " << line << " GL_INVALID_ENUM " << std::endl;
            break;
        case GL_INVALID_VALUE:
            std::cout << file << " Line: " << line << " GL_INVALID_VALUE " << std::endl;
            break;
        case GL_INVALID_OPERATION:
            std::cout << file << " Line: " << line << " GL_INVALID_OPERATION " << std::endl;
            break;
        case GL_STACK_OVERFLOW:
            std::cout << file << " Line: " << line << " STACK_OVERFLOW " << std::endl;
            break;
        case GL_STACK_UNDERFLOW:
            std::cout << file << " Line: " << line << " STACK_UNDERFLOW " << std::endl;
            break;
        case GL_OUT_OF_MEMORY:
            std::cout << file << " Line: " << line << " GL_OUT_OF_MEMORY " << std::endl;
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            std::cout << file << " Line: " << line << " GL_INVALID_FRAMEBUFFER_OPERATION " << std::endl;
            break;
        //case GL_CONTEXT_LOST:
            //std::cout << file << " Line: " << line << " GL_CONTEXT_LOST " << std::endl;
            //break;
        case GL_TABLE_TOO_LARGE:
            std::cout << file << " Line: " << line << " GL_TABLE_TOO_LARGE " << std::endl;
            break;
        default:
            std::cout << file << " Line: " << line << " GL_ERROR: " << curr_err << std::endl;
        }
        prev_err = curr_err;
    }
    return curr_err;
}

/**
 * @brief VBOHandle::init
 * Set Default member values here.
 */
template<typename T>
void VBOHandle<T>::init()
{
    m_opengl = 0;
    m_comp = 3;
    m_offset = 0;
    m_stride = 0;
    m_glsl = -1;
    m_type = GL_FLOAT;
    m_target = GL_ARRAY_BUFFER;
}

template<typename T>
VBOHandle<T>::VBOHandle()
{
    init();
}

template<typename T>
void VBOHandle<T>::Generate()
{
    if( m_opengl != 0 )
    {
        glDeleteBuffers( 1, &m_opengl );
        m_opengl = 0;
    }

    glGenBuffers( 1, &m_opengl );
}

template<typename T>
void VBOHandle<T>::Destroy()
{
    if( m_opengl != 0 )
    {
        glDeleteBuffers( 1, &m_opengl );
        m_opengl = 0;
    }

    init();
}

template<typename T>
void VBOHandle<T>::TransferToGPU()
{
    if( m_opengl != 0 )
    {
        glBindBuffer( m_target, m_opengl );
        glBufferData( m_target, m_data.size()*sizeof(T), &(m_data[0]), GL_STATIC_DRAW );
    }
}

/**
 * @brief VBOHandle::EnableUsage
 * Call this function before using it in a shader or opengl program.
 */
template<typename T>
void VBOHandle<T>::Enable(int glsl_handle)
{
    if( glsl_handle >=0 && m_target == GL_ARRAY_BUFFER )
    {
        MacroglErrorCheck();
        glEnableVertexAttribArray( glsl_handle );
        MacroglErrorCheck();
        glBindBuffer( m_target, m_opengl ); // maybe this step is not required.
        MacroglErrorCheck();
        glVertexAttribPointer( glsl_handle, m_comp, m_type, GL_FALSE, m_stride, reinterpret_cast<void*>(m_offset) );
        MacroglErrorCheck();
        m_glsl = glsl_handle;
    }
    else
        MacroWarning("Vertex Buffer Object not valid.");
}

template<typename T>
void VBOHandle<T>::Disable()
{
    if(m_glsl >= 0)
        glDisableVertexAttribArray( m_glsl );

    m_glsl = -1;
    MacroglErrorCheck();
}

void AttribHandle::init()
{
    m_glsl = -1;
}

AttribHandle::AttribHandle()
{
    init();
}
/**
 * @brief AttribHandle::GLSLBindAttribute
 * @param program Shader program handle to which the variable must be bound.
 * @param attrib_name Name of the variable used in the shader program.
 */
void AttribHandle::GLSLBindAttribute( GLuint program, const char* attrib_name )
{
    m_glsl = glGetAttribLocation( program, attrib_name );
    if( m_glsl == -1 )
    {
        MacroWarning( "Could not bind attribute:" << attrib_name );
    }
}

// Explicit Instantiation
template class VBOHandle<GLfloat>;
template class VBOHandle<GLuint>;







