#ifndef VPTEXTURECONTEXT_H
#define VPTEXTURECONTEXT_H

#include <cstdlib>
#include <stddef.h>
#include <QOpenGLWidget>
#include "vpTypes.h"
#include "vpStructures.h"
#include "vpCommon.h"

template <typename T> class vpVolume;

class LIBRARY_API vpTextureContext : public QOpenGLWidget
{
public:
    vpTextureContext(QWidget* parent = 0);
    ~vpTextureContext();

    int Refresh();
    void UpdateSegmentationTexture();

    MacroGetMember( TextureHandle, m_vol_tex,  VolumeTexture   )
    MacroGetMember( TextureHandle, m_grad_tex, GradientTexture )
    MacroGetMember( TextureHandle, m_seg_tex,  SegTexture      )
    MacroGetMember( TextureHandle, m_panc_tex, PancreasTexture )
    MacroGetMember( TextureHandle, m_cyst_tex, CystTexture )
    MacroGetMember( TextureHandle, m_duct_tex, DuctTexture )
    MacroSetMember( const vpVolume<short>*, m_vol, VolumeData )
    MacroSetMember( const vpVolume<segType>*, m_seg, SegData    )
    MacroGetSetMember( bool, m_flag_vol,       CreateVolumeTexture   )
    MacroGetSetMember( bool, m_flag_seg,       CreateSegTexture      )
    MacroGetSetMember( bool, m_flag_grad,      CreateGradientTexture )

    int CreatePancreasVolume();
    int CreateCystVolume();
    int CreateDuctVolume();
    int GetPancBoxInTexCoord(float box[6]);
    void SetPancreasBBox(size_t bbox[6]);

protected:
    virtual void initializeGL();

private:
    const vpVolume<short>* m_vol;
    const vpVolume<segType>* m_seg;
    vpVolume<float> *m_panc, *m_cyst, *m_duct;
    TextureHandle m_vol_tex, m_seg_tex, m_grad_tex, m_panc_tex, m_cyst_tex, m_duct_tex;
    bool m_flag_vol, m_flag_seg, m_flag_grad;
    size_t m_pancreas_box[6];

    void init();
    void convert_to_pixeltype ( vpn::volPixelType* pixelData );
    void convert_to_non_neg   ( short int *pixelData );
    void create_seg_texture   ( size_t dim[3], const segType *segLabelData         );

    void create_volume_texture( size_t dim[3], const vpn::volPixelType* volumeData );
    void create_volume_texture( size_t dim[3], const short* volumeData );

    void create_gradient_texture( size_t dim[3], const char* gradients  );
    int compute_pancreas_box();
    int create_label_volume( vpVolume<float>** newVol, ushort segLab );
    void create_label_texture( TextureHandle& tex, size_t dim[3], const float* data );
    void compute_gradients(vpn::volPixelType *pixelData, char* gradients);
};

#endif // VPTEXTURECONTEXT_H
