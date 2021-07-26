#ifdef _WIN32
    #include <Windows.h>
#endif

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QOpenGLFunctions_4_0_Core>

#include "TextureContext.h"
#include "ds/Image.h"

#define HSEGDEF_MAX_TEXTURE_DIM 1024

using sjDS::Image;
using sjDS::Grid;

void TextureContext::init()
{
    setVisible(false);
}

TextureContext::TextureContext(QWidget* parent)
    : QOpenGLWidget(parent)
{
    init();
}

TextureContext::~TextureContext()
{
    makeCurrent();
    m_vol.rTex().Destroy();
    m_vol_slices.rTex().Destroy();
    m_seg.rTex().Destroy();
    m_seg_slices.rTex().Destroy();
    m_grad_slices.rTex().Destroy();
}

void TextureContext::SetImage(const Image *img)
{
    NullCheckVoid(img);
    m_vol.SetData( img );
    m_vol_slices.SetData( img );
}

void TextureContext::SetFloatImage(const float* img, const Grid& grid)
{
    NullCheckVoid(img);
    m_vol.SetFloatData( img );
    m_vol.SetGrid( grid );
    m_vol_slices.SetFloatData( img );
    m_vol_slices.SetGrid( grid );
}

void TextureContext::SetGradients(const float *grads, const Grid& grid)
{
    NullCheckVoid( grads );
    m_grad_slices.SetFloatData( grads );
    m_grad_slices.SetGrid( grid );
}

void TextureContext::SetSeg(const Image *seg)
{
    NullCheckVoid(seg);
    m_seg.SetData( seg );
    m_seg_slices.SetData( seg );
}

void TextureContext::SetOpticalProperties( const Image* optical)
{
    NullCheckVoid(optical);
    m_optical.SetData( optical );
}

int TextureContext::create_texture_array( const size_t dim[3], const type_uint* volumeData, TextureHandle& tex_hdl )
{
    NullCheck(dim, 0);
    NullCheck(volumeData, 0);

    makeCurrent();
    if (tex_hdl.opengl_tex == 0)
    {
        //glPixelStorei( GL_UNPACK_ALIGNMENT, 4 );
        glGenTextures(1, &tex_hdl.opengl_tex);

        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_hdl.opengl_tex);

        //glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        //glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
        //glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        //glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER );
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32UI, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED_INTEGER, GL_UNSIGNED_INT, volumeData);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_hdl.opengl_tex);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32UI, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED_INTEGER, GL_UNSIGNED_INT, volumeData);
    }

    GLenum err = MacroglErrorCheck();

    if( err == GL_NO_ERROR )
        return 1;
    else
        return 0;
}

int TextureContext::create_1dtexture_array( size_t dim_x, size_t dim_y, const type_uint* data, TextureHandle& tex_hdl )
{
    if( dim_x < 1 || dim_y < 1 )
        return 0;

    NullCheck(data, 0);

    makeCurrent();

    glGenTextures( 1, &tex_hdl.opengl_tex );
    glBindTexture( GL_TEXTURE_1D_ARRAY, tex_hdl.opengl_tex );
    glTexParameteri( GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_1D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );

    glTexImage2D( GL_TEXTURE_1D_ARRAY, 0, GL_RGBA, (GLsizei)dim_x, (GLsizei)dim_y,
                      0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, (const void*)data );

    GLenum err = MacroglErrorCheck();

    if( err == GL_NO_ERROR )
        return 1;
    else
        return 0;
}


int TextureContext::create_texture_array( const size_t dim[3], const float* volumeData, TextureHandle& tex_hdl )
{
    NullCheck(dim, 0);
    NullCheck(volumeData, 0);

    makeCurrent();

    if (tex_hdl.opengl_tex == 0)
    {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glGenTextures(1, &tex_hdl.opengl_tex);
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_hdl.opengl_tex);

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED, GL_FLOAT, volumeData);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_hdl.opengl_tex);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED, GL_FLOAT, volumeData);
    }

    GLenum err = MacroglErrorCheck();

    if( err == GL_NO_ERROR )
        return 1;
    else
        return 0;
}

int TextureContext::create_volume_texture(const size_t dim[3], const float* volumeData, TextureHandle& tex_hdl )
{
    NullCheck(dim, 0);
    NullCheck(volumeData, 0);

    makeCurrent();
    MacroglErrorCheck();
    if (tex_hdl.opengl_tex == 0)
    {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        MacroglErrorCheck();
        glGenTextures(1, &tex_hdl.opengl_tex);
        MacroglErrorCheck();
        glBindTexture(GL_TEXTURE_3D, tex_hdl.opengl_tex);
        MacroglErrorCheck();
        //glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

        MacroglErrorCheck();
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED, GL_FLOAT, volumeData);
    }
    else
    {
        MacroglErrorCheck();
        glBindTexture(GL_TEXTURE_3D, tex_hdl.opengl_tex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED, GL_FLOAT, volumeData);
    }

    GLenum r = MacroglErrorCheck();
    if( r == GL_NO_ERROR )
        return 1;
    else
        return 0;
}

int TextureContext::create_volume_texture(const size_t dim[3], const type_uint* volumeData, TextureHandle& tex_hdl )
{
    NullCheck(dim, 0);
    NullCheck(volumeData, 0);

    makeCurrent();
    MacroglErrorCheck();

    if (tex_hdl.opengl_tex == 0)
    {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glGenTextures(1, &tex_hdl.opengl_tex);
        glBindTexture(GL_TEXTURE_3D, tex_hdl.opengl_tex);
        //glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

        MacroglErrorCheck();
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED_INTEGER, GL_UNSIGNED_INT, volumeData);
    }
    else
    {
        MacroglErrorCheck();
        glBindTexture(GL_TEXTURE_3D, tex_hdl.opengl_tex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED_INTEGER, GL_UNSIGNED_INT, volumeData);
    }

    GLenum r = MacroglErrorCheck();
    if( r == GL_NO_ERROR )
        return 1;
    else
        return 0;
}

int TextureContext::create_2d_texture(size_t dim[3], const type_uint* data, TextureHandle& tex_hdl )
{
    NullCheck(data, 0);
    //int* x = new int[100];
    //std::cout << x << std::endl;

    size_t tex_dim[2] = {0,0};
    size_t max_dim = HSEGDEF_MAX_TEXTURE_DIM;
    tex_dim[1] =  dim[0] < max_dim ? size_t(1) : size_t(dim[0] / max_dim + 1);
    tex_dim[0] = max_dim;

    // Copy data to new raw array so correct for slight size mismatch due to rounding-off
    // of 2D Texture Dimensions.
    const size_t array_sz = tex_dim[0] * tex_dim[1];
    type_uint* resized_data = new type_uint[ array_sz ];
    memcpy( (void*)resized_data, (void*)data, sizeof(type_uint)* dim[0] );

    // Switch to OpenGL context of current object.
    makeCurrent();

    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &tex_hdl.opengl_tex );
    glBindTexture( GL_TEXTURE_2D, tex_hdl.opengl_tex );
//    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    //glTexImage1D( GL_TEXTURE_1D, 0, GL_RGBA8UI, dim[0], 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT_8_8_8_8, data );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8UI, (GLsizei)tex_dim[0], (GLsizei)tex_dim[1], 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT_8_8_8_8, resized_data );

    MacroDeleteArray(resized_data);

    GLenum r = MacroglErrorCheck();
    if( r == GL_NO_ERROR )
        return 1;
    else
        return 0;
}

int TextureContext::Refresh()
{
    //MacroConfirmOrReturn( m_vol.Data(), 0 );
    //MacroConfirmOrReturn( m_vol_slices.Data(), 0 );

    //int ret1 = 1, ret2 = 1, ret3 = 1;
    int ret = 1;

    makeCurrent();

    // Construct 3D volume texture out of the input volume of uint or float type.
    if( m_vol.Data() && m_vol.Outdated() )
    {
        if(m_vol.SizeChanged())
            m_vol.rTex().Destroy();

        ret &= update_texture_from_image( m_vol, true, false );
        m_vol.SetUpdated();
    }
    else if( m_vol.FloatData() && m_vol.Outdated() )
    {
        if(m_vol.SizeChanged())
            m_vol.rTex().Destroy();

        ret &= create_volume_texture( (size_t*)m_vol.Grid().Dim(), (const float*)m_vol.FloatData(), m_vol.rTex() );
        m_vol.SetUpdated();
    }

    // Construct array of 2D textures out of the input volume of uint or float type.
    if( m_vol_slices.Data() && m_vol_slices.Outdated() )
    {
        if(m_vol_slices.SizeChanged())
            m_vol_slices.rTex().Destroy();

        ret &= update_texture_from_image( m_vol_slices, true, true );
        m_vol_slices.SetUpdated();
    }
    else if(m_vol_slices.FloatData() && m_vol_slices.Outdated() )
    {
        if(m_vol_slices.SizeChanged())
            m_vol_slices.rTex().Destroy();
        ret &= create_texture_array( m_vol_slices.Grid().Dim(), m_vol_slices.FloatData(), m_vol_slices.rTex() );
        m_vol_slices.SetUpdated();
    }

    // Create array of 2D textures out of gradients
    if( m_grad_slices.FloatData() && m_grad_slices.Outdated() )
    {
        //if(m_grad_slices.SizeChanged())
            m_grad_slices.rTex().Destroy();

        ret &= update_gradients_texture( m_grad_slices );
        m_grad_slices.SetUpdated();
    }

    // Create 3D texture out of segmentation mask
    if( m_seg.Data() && m_seg.Outdated() )
    {
        if(m_seg.SizeChanged())
            m_seg.rTex().Destroy();
        ret &= update_texture_from_image( m_seg, false, false );
        m_seg.SetUpdated();
    }

    // Create array of 2D textures out of segmentation mask
    if( m_seg_slices.Data() && m_seg_slices.Outdated() )
    {
        if(m_seg_slices.SizeChanged())
            m_seg_slices.rTex().Destroy();
        ret &= update_texture_from_image( m_seg_slices, false, true );
        m_seg_slices.SetUpdated();
    }

    // Create array of 1D textures from sampled transfer functions of each segment.
    if( m_optical.Data() && m_optical.Outdated() )
    {
        //if(m_optical.SizeChanged())
            m_optical.rTex().Destroy();

        size_t dim[3]={0,0,0};
        const type_uint* pixelData = m_optical.Data()->GetDataPointer();
        m_optical.Data()->GetDimensions(dim);
        ret &= create_1dtexture_array( dim[0], dim[1], pixelData, m_optical.rTex() );
        m_optical.SetUpdated();
    }

    if( ret )
        return 1;
    else
        return 0;
}

int TextureContext::update_texture_from_image( TextureContext::TexObject& tex_obj,
                                               bool float_conv, bool array_texture )
{
    NullCheck( tex_obj.Data(), 0 );

    size_t sz = tex_obj.Data()->GetArraySize();
    size_t dim[3]={0,0,0};
    tex_obj.Data()->GetDimensions(dim);

    int ret = 0;
    if( float_conv )
    {
        //float* pixelData = new float[sz];
        std::vector<float> pixelData(sz);
        //memset( pixelData, 0, sz*sizeof(float) );
        convert_to_pixeltype( pixelData.data(), tex_obj.Data() );
        if( array_texture )
            ret = create_texture_array( dim, pixelData.data(), tex_obj.rTex() );
        else
            ret = create_volume_texture( dim, pixelData.data(), tex_obj.rTex() );

        //MacroDeleteArray(pixelData);
    }
    else
    {
        const type_uint* pixelData = tex_obj.Data()->GetDataPointer();
        if( array_texture )
            ret = create_texture_array( dim, pixelData, tex_obj.rTex() );
        else if( tex_obj.Data()->is3D() )
            ret = create_volume_texture( dim, pixelData, tex_obj.rTex() );
        else
            ret = create_2d_texture( dim, pixelData, tex_obj.rTex() );
    }

    return ret;
}

// Convert voxel values to normalized floats.
void TextureContext::convert_to_pixeltype( float* pixelData, const Image* img )
{
    NullCheckVoid(img);
    const type_uint* array_ptr = img->GetDataPointer();
    NullCheckVoid(array_ptr);

    size_t dim[3]={0,0,0};
    img->GetDimensions(dim);
    type_uint range[2] = {0,0};
    img->GetScalarRange( range );
    MacroAssert( range[1] >= range[0] );

    //type_uint rangeDiff = std::max(range[1],range[0]) - std::min(range[1],range[0]);
    type_uint rangeDiff = range[1] - range[0];

    type_uint unsigned_range = rangeDiff+1;

    size_t array_size = img->GetArraySize();
    for( size_t i=0; i < array_size; i++)
    {
        float hu_val = (float)array_ptr[i];
        hu_val = ( hu_val-range[0] ) / float(unsigned_range);
        MacroAssert(hu_val >= 0);
        pixelData[i] = hu_val;
    }
}

int TextureContext::UpdateOpticalTexture()
{
    int ret = 0;
    NullCheck(m_optical.Data(), 0);
    makeCurrent();
    TextureHandle& tex_hdl = m_optical.rTex();

    if( tex_hdl.opengl_tex )
    {
        size_t dim[3]={0,0,0};
        m_optical.Data()->GetDimensions(dim);
        glBindTexture( GL_TEXTURE_1D_ARRAY, tex_hdl.opengl_tex );
        //glTexParameteri( GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
        //glTexParameteri( GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        //glTexParameteri( GL_TEXTURE_1D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );

        glTexImage2D( GL_TEXTURE_1D_ARRAY, 0, GL_RGBA, (GLsizei)dim[0], (GLsizei)dim[1],
                          0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, (const void*)m_optical.Data()->GetDataPointer() );

        GLenum err = MacroglErrorCheck();

        if( err == GL_NO_ERROR )
            ret = 1;
    }
    else
    {
        MacroWarning("Optical 1D Texture Array not defined.");
    }

    m_optical.SetUpdated();
    return ret;
}

int TextureContext::UpdateSegTexture()
{
    makeCurrent();
    MacroAssert(m_seg_slices.Data() );

    size_t dim[3]={0,0,0};
    m_seg_slices.Data()->GetDimensions(dim);

    int ret = 1;
    if(m_seg_slices.rTex().opengl_tex )
    {
        glBindTexture( GL_TEXTURE_2D_ARRAY,m_seg_slices.rTex().opengl_tex  );
        glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, GL_R32UI, (GLsizei)dim[0], (GLsizei)dim[1],
                      (GLsizei)dim[2], 0, GL_RED_INTEGER, GL_UNSIGNED_INT, m_seg_slices.Data()->GetDataPointer() );

        GLenum r = MacroglErrorCheck();
        if( r != GL_NO_ERROR )
           ret = 0;
    }
    else
    {
        MacroWarning("Segment TextureArray not defined.");
        ret = 0;
    }

    if(m_seg.rTex().opengl_tex )
    {
        glBindTexture( GL_TEXTURE_3D, m_seg.rTex().opengl_tex  );
        glTexImage3D( GL_TEXTURE_3D, 0, GL_R32UI, (GLsizei)dim[0], (GLsizei)dim[1],
                      (GLsizei)dim[2], 0, GL_RED_INTEGER, GL_UNSIGNED_INT, m_seg.Data()->GetDataPointer() );

        GLenum r = MacroglErrorCheck();
        if( r != GL_NO_ERROR )
           ret = 0;
    }
    else
    {
        MacroWarning("Segment Texture not defined.");
        ret = 0;
    }

    m_seg_slices.SetUpdated();
    m_seg.SetUpdated();
    return ret;
}

int TextureContext::UpdateSegTexture(const size_t dim[3], const float* volumeData, TextureHandle& tex_hdl )
{
    NullCheck(dim, 0);
    NullCheck(volumeData, 0);

    makeCurrent();
    MacroglErrorCheck();
    if (tex_hdl.opengl_tex == 0)
    {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glGenTextures(1, &tex_hdl.opengl_tex);
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_hdl.opengl_tex);

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        MacroglErrorCheck();
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED, GL_FLOAT, volumeData);
    }
    else
    {
        MacroglErrorCheck();
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_hdl.opengl_tex);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED, GL_FLOAT, volumeData);
    }

    GLenum r = MacroglErrorCheck();
    if( r == GL_NO_ERROR )
        return 1;
    else
        return 0;
}

int TextureContext::UpdateSegTextureArray(const size_t dim[3], const uint* volumeData, TextureHandle& tex_hdl )
{
    NullCheck(dim, 0);
    NullCheck(volumeData, 0);

    makeCurrent();
    MacroglErrorCheck();
    if (tex_hdl.opengl_tex == 0)
    {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        MacroglErrorCheck();
        glGenTextures(1, &tex_hdl.opengl_tex);
        MacroglErrorCheck();
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_hdl.opengl_tex);
        MacroglErrorCheck();
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        MacroglErrorCheck();
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        MacroglErrorCheck();
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        MacroglErrorCheck();
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        MacroglErrorCheck();
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        MacroglErrorCheck();
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32UI, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED_INTEGER, GL_UNSIGNED_INT, volumeData);
        MacroglErrorCheck();
    }
    else
    {
        MacroglErrorCheck();
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_hdl.opengl_tex);
        MacroglErrorCheck();
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32UI, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED_INTEGER, GL_UNSIGNED_INT, volumeData);
        MacroglErrorCheck();
    }

    GLenum r = MacroglErrorCheck();
    if( r == GL_NO_ERROR )
        return 1;
    else
        return 0;
}

void TextureContext::initializeGL()
{
    makeCurrent();
    /*
    GLenum err = glewInit();

    if( err != GLEW_OK )
    {
        MacroWarning("Glew init failed.");
        return;
    }
    */
}

int TextureContext::update_gradients_texture( TextureContext::TexObject& tex_obj )
{
    NullCheck( tex_obj.FloatData(), 0 );

    size_t dim[3]={0,0,0};
    tex_obj.Grid().GetDimensions(dim);
    dim[0] = dim[0]/3;

    int ret = 0;

    makeCurrent();
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &tex_obj.rTex().opengl_tex );

    glBindTexture( GL_TEXTURE_2D_ARRAY, tex_obj.rTex().opengl_tex );

    glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );

    glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, GL_RGB32F, (GLsizei)dim[0], (GLsizei)dim[1],
                      (GLsizei)dim[2], 0, GL_RGB, GL_FLOAT, (const void*)tex_obj.Data() );

    GLenum err = MacroglErrorCheck();

    if( err == GL_NO_ERROR )
        return 1;
    else
        return 0;

    return ret;
}
