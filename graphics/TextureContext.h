#ifndef TEXTURE_CONTEXT_H
#define TEXTURE_CONTEXT_H

#include <cstdlib>
#include <stddef.h>
#include <QOpenGLWidget>
#include "core/CoreTypedefs.h"
#include "ds/Grid.h"
#include "ds/Image.h"
#include "vpStructures.h"


class QOpenGLFunctions;
class QOpenGLFunctions_4_0_Core;

class TextureContext : public QOpenGLWidget
{
public:
    static const uint sOpticalTextureResolution = 256;
    TextureContext(QWidget* parent = 0);

    ~TextureContext();

    int Refresh();

    void SetImage            ( const sjDS::Image* img     );
    void SetFloatImage(const float* img, const sjDS::Grid& grid);
    void SetGradients(const float *grads, const sjDS::Grid& grid);
    void   SetSeg            ( const sjDS::Image* seg     );
    void SetOpticalProperties( const sjDS::Image* optical );
    int UpdateOpticalTexture();
    int UpdateSegTexture();
    int UpdateSegTexture(const size_t dim[3], const float* volumeData, TextureHandle& tex_hdl);
    int UpdateSegTextureArray(const size_t dim[3], const uint* volumeData, TextureHandle& tex_hdl);

    MacroGetMember( TextureHandle, m_vol.rTex(), VolumeTexture   )
    MacroGetMember( TextureHandle, m_vol_slices.rTex(), VolumeTextureArray )
    MacroGetMember( TextureHandle, m_seg.rTex(), SegmentTexture  )
    MacroGetMember( TextureHandle, m_seg_slices.rTex(), SegmentTextureArray  )
    MacroGetMember( TextureHandle, m_optical.rTex(), OpticalTexture  )
    MacroGetMember( TextureHandle, m_grad_slices.rTex(), GradientTextureArray )

protected:
    virtual void initializeGL();

private:
    struct TexObject;

    void  init();
    int   create_volume_texture( const size_t dim[3], const float* volumeData, TextureHandle& tex_hdl );
    int   create_volume_texture( const size_t dim[3], const type_uint* volumeData, TextureHandle& tex_hdl );
    int   create_texture_array ( const size_t dim[3], const float* volumeData, TextureHandle& tex_hdl );
    int   create_texture_array ( const size_t dim[3], const type_uint* volumeData, TextureHandle& tex_hdl );
    int   create_1dtexture_array( size_t dim_x, size_t dim_y, const type_uint* volumeData, TextureHandle& tex_hdl );
    int   create_2d_texture    ( size_t dim[3], const type_uint* data, TextureHandle& tex_hdl );
    void  convert_to_pixeltype( float* pixelData, const sjDS::Image* img );
    int   update_texture_from_image( TexObject& tex_obj, bool float_conv, bool array_texture );
    int update_gradients_texture( TextureContext::TexObject& tex_obj );

    struct TexObject
    {
    private:
        const sjDS::Image*  m_data;
        const float* m_float_data;
        sjDS::Grid m_grid;
        TextureHandle       m_tex_hdl;
        bool                m_out_dated;
        bool                m_size_change;

    public:
        TexObject()
        {
            m_data = NULL;
            m_float_data = NULL;
            m_out_dated = true;
            m_size_change = false;
        }

        TextureHandle& rTex()
        {
            return m_tex_hdl;
        }

        const TextureHandle& rTex() const
        {
            return m_tex_hdl;
        }

        const sjDS::Image* Data() const
        {
            return m_data;
        }
        const float* FloatData() const
        {
            return m_float_data;
        }

        const sjDS::Grid& Grid() const
        {
            return m_grid;
        }

        bool Outdated() const
        {
            return m_out_dated;
        }

        bool SizeChanged() const
        {
            return m_size_change;
        }

        void SetData( const sjDS::Image* in )
        {
            if(m_data && in)
                m_size_change = m_data->isSameDim(*in);

            m_data = in;
            SetOutdated();
        }

        void SetFloatData( const float* data)
        {
            m_float_data = data;
            SetOutdated();
        }

        void SetGrid( const sjDS::Grid& grid)
        {
            m_size_change = (m_grid.x() == grid.x() && m_grid.y() == grid.y() && m_grid.z() == grid.z());
            m_grid = grid;
            SetOutdated();
        }

        void SetOutdated()
        {
            m_out_dated = true;
        }

        void SetUpdated()
        {
            m_out_dated = false;
        }
    };

    TexObject m_vol, m_vol_slices, m_seg, m_seg_slices, m_optical, m_grad_slices;
};

#endif // TEXTURE_CONTEXT_H
