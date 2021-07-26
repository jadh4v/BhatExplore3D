#ifndef VPSTRUCTURES
#define VPSTRUCTURES

#include <vector>
#include <GL/gl.h>

#include "core/macros.h"
#include "vpCommon.h"

#define MacroglErrorCheck() glErrorCheck(__FILE__,__LINE__)

GLenum glErrorCheck( const char* file, const int line );

typedef unsigned int uint;

struct LIBRARY_API TextureHandle
{
public:
    uint opengl_tex;
    int glsl_tex;

    TextureHandle()
    {
        opengl_tex = 0; glsl_tex = 0;
    }

    void Destroy()
    {
        if( opengl_tex != 0)
        {
            glDeleteTextures( 1, &opengl_tex );
            opengl_tex = 0;
        }
    }
};

template<typename T>
class VBOHandle
{
private:
    uint m_opengl;
    GLenum m_type, m_target;
    int  m_comp, m_offset, m_stride, m_glsl;

    void init();

public:
    std::vector<T> m_data;

    // RESOURCE Management
    VBOHandle();
    void Generate();
    void Destroy();

    void Enable(int glsl_handle);
    void Disable();

    //GET
    uint   GetOpenGLHandle()  const { return m_opengl; }
    GLenum GetType()          const { return m_type;   }
    int  GetNumComponents() const { return m_comp;   }
    int  GetOffset()        const { return m_offset; }
    int  GetStride()        const { return m_stride; }
    GLenum GetTarget()      const { return m_target; }

    //SET
    void SetType(GLenum a)  { m_type = a; }
    void SetNumComponents( int a ) { m_comp   = a; }
    void SetOffset( int a )        { m_offset = a*sizeof(GLfloat); }
    void SetStride( int a )        { m_stride = a*sizeof(GLfloat); }
    void SetTarget(GLenum target)  { m_target = target; }

    // OPERATION
    void TransferToGPU();
};

struct AttribHandle
{
private:
    int  m_glsl;

    void init();

public:
    AttribHandle();
    int  GetGLSLHandle()    const { return m_glsl;   }
    void GLSLBindAttribute( GLuint program, const char* attrib_name );
};

#endif // VPSTRUCTURES
