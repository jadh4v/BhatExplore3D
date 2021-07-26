#include <cmath>
#include <GL/glew.h>

#include <QTime>

#include "vpVolume.h"
#include "vpTextureContext.h"
//#include "vpVolAlgoSmooth.h"

#define LABEL_PANCREAS 255
#define LABEL_DUCT 255
#define LABEL_CYST 255
/*
#define LABEL_PANCREAS 4000
#define LABEL_DUCT 5000
#define LABEL_CYST 6000
*/

void vpTextureContext::init()
{
    m_vol  = NULL;
    m_seg  = NULL;
    m_panc = m_cyst = m_duct = NULL;
    m_flag_grad = m_flag_vol = m_flag_seg = false;
    memset( m_pancreas_box, 0, 6*sizeof(size_t));
    setVisible(false);
}

vpTextureContext::vpTextureContext(QWidget* parent)
    : QOpenGLWidget(parent)
{
    init();
    setFixedSize(1, 1);
}

vpTextureContext::~vpTextureContext()
{
    glDeleteTextures( 1, &m_panc_tex.opengl_tex );
    glDeleteTextures( 1, &m_cyst_tex.opengl_tex );
    glDeleteTextures( 1, &m_duct_tex.opengl_tex );
}

void vpTextureContext::initializeGL()
{
    makeCurrent();
    std::cout << "vpTextureContext::initializeGL() " << this->context() << std::endl;
    // Initial setup rendering cgContext:
    glClearColor( 0.0f, 0.0f, 0.0f, 1.0f ); // set background color
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glEnable(GL_DEPTH_TEST);
}

int vpTextureContext::Refresh()
{
    makeCurrent();
    if (this->context() == nullptr)
    {
        MacroWarning("OpenGL context is not initialized yet. Unable to construct textures.");
        return 0;
    }
    if( GetCreateVolumeTexture() )
    {
        MacroConfirmOrReturn( m_vol, 0 );

        size_t sz = m_vol->GetArraySize();
        vpn::volPixelType* pixelData = new vpn::volPixelType[sz];
        memset( pixelData, 0, sz*sizeof(vpn::volPixelType) );
        QTime t;
        t.start();
        convert_to_pixeltype( pixelData );
        float sec = t.elapsed() / 1000.0f;
        MacroMessage(" Time for dataset rescaling: "
                     << sec << " sec. ");
//        short int* pixelData = new short int[sz];
//        memset( pixelData, 0, sz*sizeof(short int) );
//        convert_to_non_neg( pixelData );

        size_t dim[3]={0,0,0};
        m_vol->GetDimensions(dim);
        create_volume_texture( dim, pixelData );
//        create_volume_texture( dim, m_vol->GetDataPointer() );

#ifdef COMPUTED_GRADIENTS
        char* gradients = new char[sz*3];
        memset( gradients, 0, 3*sz*sizeof(char) );
        compute_gradients( pixelData, gradients );
        create_gradient_texture( dim, gradients );
        MacroDeleteArray( gradients );
#endif

        //MacroDeleteArray( pixelData );

    }

    if( GetCreateSegTexture() )
    {
        MacroConfirmOrReturn( m_seg, 0 );
        size_t dim[3]={0,0,0};
        m_seg->GetDimensions(dim);
        create_seg_texture( dim, m_seg->GetDataPointer() );
    }

    /* TODO: compute gradients in this class first.
    if( GetCreateGradientTexture() )
    {
        MacroConfirmOrReturn( m_grad, 0 );
        size_t dim[3]={0,0,0};
        m_grad->GetDimensions(dim);
        create_seg_texture( dim, pixelData );
    }
    */

    return 1;
}

int vpTextureContext::CreatePancreasVolume()
{
    // Create smaller texture using bounding box for pancreas labels.
    int succ = create_label_volume(&m_panc, LABEL_PANCREAS);
    if( !succ )
        return 0;

    size_t pandim[3]={0,0,0};
    m_panc->GetDimensions(pandim);
    create_label_texture( m_panc_tex, pandim, m_panc->GetDataPointer() );
    return 1;
}
int vpTextureContext::CreateCystVolume()
{
    // Create smaller texture using bounding box for pancreas labels.
    int succ = create_label_volume(&m_cyst, LABEL_CYST);
    if( !succ )
        return 0;

    size_t pandim[3]={0,0,0};
    m_cyst->GetDimensions(pandim);
    create_label_texture( m_cyst_tex, pandim, m_cyst->GetDataPointer() );
    return 1;
}
int vpTextureContext::CreateDuctVolume()
{
    // Create smaller texture using bounding box for pancreas labels.
    int succ = create_label_volume(&m_duct, LABEL_DUCT);
    if( !succ )
        return 0;

    size_t pandim[3]={0,0,0};
    m_duct->GetDimensions(pandim);
    create_label_texture( m_duct_tex, pandim, m_duct->GetDataPointer() );
    return 1;
}

void vpTextureContext::create_volume_texture(size_t dim[3], const vpn::volPixelType* volumeData )
{
    //glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &m_vol_tex.opengl_tex );
    glBindTexture( GL_TEXTURE_3D, m_vol_tex.opengl_tex );
    //glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    //MacroglErrorCheck();
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    glTexImage3D( GL_TEXTURE_3D, 0, GL_R16F, (GLsizei)dim[0], (GLsizei)dim[1],
                  (GLsizei)dim[2], 0, GL_RED, GL_FLOAT, volumeData );

    MacroglErrorCheck();
}
void vpTextureContext::create_volume_texture(size_t dim[3], const short* volumeData )
{
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &m_vol_tex.opengl_tex );
    glBindTexture( GL_TEXTURE_3D, m_vol_tex.opengl_tex );
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER );

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    //glPixelTransferf(GL_RED_SCALE, (1.0f/4096.0f));
    glPixelTransferf(GL_RED_SCALE, (16.0f));
    glPixelTransferf(GL_RED_BIAS, 0.0f);
//    glPixelTransferf(GL_RED_SCALE, (8.0f));
//    glPixelTransferf(GL_RED_BIAS, 0.25f);

    glTexImage3D( GL_TEXTURE_3D, 0, GL_R16F, (GLsizei)dim[0], (GLsizei)dim[1],
                  (GLsizei)dim[2], 0, GL_RED, GL_SHORT, volumeData );

    MacroglErrorCheck();
    glPixelTransferf(GL_RED_SCALE, 1.0f);
    glPixelTransferf(GL_RED_BIAS, 0.0f);
}

void vpTextureContext::create_seg_texture( size_t dim[3], const segType* segLabelData )
{
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &m_seg_tex.opengl_tex );
    glBindTexture( GL_TEXTURE_3D, m_seg_tex.opengl_tex );
    //glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
//    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
//    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER );

//    glTexImage3D( GL_TEXTURE_3D, 0, GL_R8UI, dim[0], dim[1], dim[2],
//                  0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, segLabelData );
    glTexImage3D( GL_TEXTURE_3D, 0, GL_R16UI, (GLsizei)dim[0], (GLsizei)dim[1], (GLsizei)dim[2],
                  0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, segLabelData );

    MacroglErrorCheck();
}

void vpTextureContext::create_gradient_texture( size_t dim[3], const char* gradients )
{
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &m_grad_tex.opengl_tex );
    glBindTexture( GL_TEXTURE_3D, m_grad_tex.opengl_tex );
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER );

    glTexImage3D( GL_TEXTURE_3D, 0, GL_RGB8_SNORM, (GLsizei)dim[0], (GLsizei)dim[1], (GLsizei)dim[2],
                  0, GL_RGB, GL_BYTE, gradients );

    MacroglErrorCheck();
}

void vpTextureContext::create_label_texture(TextureHandle& tex, size_t dim[3], const float* data)
{
    MacroglErrorCheck();
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glGenTextures( 1, &tex.opengl_tex );
    glBindTexture( GL_TEXTURE_3D, tex.opengl_tex );
   //glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER );

    glTexImage3D( GL_TEXTURE_3D, 0, GL_R16F, (GLsizei)dim[0], (GLsizei)dim[1], (GLsizei)dim[2],
                  0, GL_RED, GL_FLOAT, data );

    MacroglErrorCheck();
}

void vpTextureContext::UpdateSegmentationTexture()
{
    makeCurrent();
    MacroConfirm( m_seg );
    MacroConfirm( m_seg->GetDataPointer() );

    size_t dim[3]={0,0,0};
    m_seg->GetDimensions(dim);

    if( m_seg_tex.opengl_tex )
    {
        glBindTexture( GL_TEXTURE_3D, m_seg_tex.opengl_tex  );

//        glTexImage3D( GL_TEXTURE_3D, 0, GL_R8UI, dim[0], dim[1], dim[2],
//                      0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, m_seg->GetDataPointer() );
        glTexImage3D( GL_TEXTURE_3D, 0, GL_R16UI, (GLsizei)dim[0], (GLsizei)dim[1], (GLsizei)dim[2],
                      0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, m_seg->GetDataPointer() );

        MacroglErrorCheck();
    }
    else
        MacroWarning("Segmentation Texture not defined.");
}

void vpTextureContext::convert_to_non_neg( short int* pixelData )
{
    NullCheckVoid(m_vol);
    short int* array_ptr = m_vol->GetDataPointer();
    NullCheckVoid(array_ptr);

    size_t dim[3]={0,0,0};
    m_vol->GetDimensions(dim);
    size_t array_size = m_vol->GetArraySize();

    short shortRange[2] = {0,0};
    m_vol->GetRange( shortRange );
    if( shortRange[0] != -1024 )
        MacroWarning("Min value is not -1024.");

    // Shift origin to make all values positive.
    for( size_t i=0; i < array_size; i++)
    {
        short int val = array_ptr[i];
        val += 1024;
        if( val < 0 ) val = 0;
        pixelData[i] = val;
    }
}

void vpTextureContext::convert_to_pixeltype( vpn::volPixelType* pixelData )
{
    NullCheckVoid(m_vol);
    short int* array_ptr = m_vol->GetDataPointer();
    NullCheckVoid(array_ptr);

    size_t dim[3]={0,0,0};
    m_vol->GetDimensions(dim);
    short shortRange[2] = {0,0};
    //m_vol->GetRange( shortRange );
    //MacroMessage("dataset range = " << shortRange[0] << ", " << shortRange[1]);
    // use fixed short range for consistency across different datasets
    shortRange[0] = -1024;
    shortRange[1]  = 3072; 
    //MacroMessage("fixed range = " << shortRange[0] << ", " << shortRange[1]);
    float range[2]={ float(shortRange[0]), float(shortRange[1]) };
    int rangeDiff = range[1]-range[0];
    MacroAssert( rangeDiff > 0);
    int unsigned_range = rangeDiff+1;

    size_t array_size = m_vol->GetArraySize();
    for( size_t i=0; i < array_size; i++)
    {
        float hu_val = (float)array_ptr[i];
        hu_val = ( hu_val-range[0] ) / float(unsigned_range);
        MacroAssert(hu_val >= 0);
        pixelData[i] = hu_val;
    }
}

void vpTextureContext::compute_gradients( vpn::volPixelType* pixelData,
                                          char* gradients )
{
    NullCheckVoid( pixelData );
    NullCheckVoid( m_vol );

    size_t dim[3]={0,0,0};
    m_vol->GetDimensions(dim);

    // Construct RGB texture for the CT gradients.
    size_t slice_sz = dim[1]*dim[0];
    // skip boundary pixels.
    for( size_t k=1; k < dim[2]-1; k++)
    {
        for( size_t j=1; j < dim[1]-1; j++)
        {
            for( size_t i=1; i < dim[0]-1; i++)
            {
                int64_t xb   = (k*slice_sz + j*dim[0] + i-1);
                int64_t xf   = (k*slice_sz + j*dim[0] + i+1);

                int64_t yb   = (k*slice_sz + (j-1)*dim[0] + i);
                int64_t yf   = (k*slice_sz + (j+1)*dim[0] + i);

                int64_t zb   = ((k-1)*slice_sz + j*dim[0] + i);
                int64_t zf   = ((k+1)*slice_sz + j*dim[0] + i);

                float xb_val = pixelData[xb];
                float xf_val = pixelData[xf];

                float yb_val = pixelData[yb];
                float yf_val = pixelData[yf];

                float zb_val = pixelData[zb];
                float zf_val = pixelData[zf];

                float grad_x = xf_val-xb_val;
                float grad_y = yf_val-yb_val;
                float grad_z = zf_val-zb_val;

                float magn = sqrt( grad_x*grad_x + grad_y*grad_y + grad_z*grad_z );
                char ch_x = (char) 127*grad_x / magn;
                char ch_y = (char) 127*grad_y / magn;
                char ch_z = (char) 127*grad_z / magn;

                size_t indx = 3*(k*slice_sz + j*dim[0] + i);
                gradients[indx+0] = ch_x;
                gradients[indx+1] = ch_y;
                gradients[indx+2] = ch_z;
            }
        }
    }
}

void vpTextureContext::SetPancreasBBox(size_t bbox[6])
{
    if( !bbox ) return;
    memcpy( m_pancreas_box, bbox, sizeof(size_t)*6 );
}

int vpTextureContext::compute_pancreas_box()
{
    if( !m_seg )
    {
        MacroWarning("Cannot compute pancreas box without segmentation data.");
        return 0;
    }
    if( !m_seg->ValidDimensions() )
    {
        MacroWarning("Cannot compute pancreas box without volume dimensions.");
        return 0;
    }

    m_pancreas_box[3] = m_pancreas_box[4] = m_pancreas_box[5] = 0; // min values
    m_pancreas_box[0] = m_pancreas_box[1] = m_pancreas_box[2] = 4096; // max values
    size_t xySize = m_seg->GetXSize()*m_seg->GetYSize();

    for( size_t k=0; k < m_seg->GetZSize(); k++)
    {
        for( size_t j=0; j < m_seg->GetYSize(); j++)
        {
            for( size_t i=0; i < m_seg->GetXSize(); i++)
            {
                size_t indx = i + j*m_seg->GetXSize() + k*xySize;
                if( (*m_seg)[indx] != 0 )
                {
                    if( m_pancreas_box[0] > i )     m_pancreas_box[0] = i;  // min X
                    if( m_pancreas_box[3] < i )     m_pancreas_box[3] = i;  // max X
                    if( m_pancreas_box[1] > j )     m_pancreas_box[1] = j;  // min Y
                    if( m_pancreas_box[4] < j )     m_pancreas_box[4] = j;  // max Y
                    if( m_pancreas_box[2] > k )     m_pancreas_box[2] = k;  // min Z
                    if( m_pancreas_box[5] < k )     m_pancreas_box[5] = k;  // max Z
                }
            }
        }
    }

    /*
    m_pancreas_box[0] -= 50;
    m_pancreas_box[1] -= 50;
    m_pancreas_box[2] -= 50;

    m_pancreas_box[3] += 50;
    m_pancreas_box[4] += 50;
    m_pancreas_box[5] += 50;
    */

    return 1;
}

int vpTextureContext::create_label_volume( vpVolume<float>** newVol, ushort segLab )
{
    MacroConfirmOrReturn(m_seg,0);

    size_t segdim[3]={0,0,0};
    m_seg->GetDimensions(segdim);

    size_t dim[3]={0,0,0};
    dim[0] = m_pancreas_box[3]-m_pancreas_box[0]+1;
    dim[1] = m_pancreas_box[4]-m_pancreas_box[1]+1;
    dim[2] = m_pancreas_box[5]-m_pancreas_box[2]+1;

    size_t sz = dim[0]*dim[1]*dim[2];
    MacroDelete(*newVol);
    *newVol = new vpVolume<float>();
    (*newVol)->SetDimensions(dim);
    (*newVol)->Initialize(sz);

    for(size_t k=0;k<dim[2];k++)
    {
        size_t ko = m_pancreas_box[2]+k;
        for(size_t j=0;j<dim[1];j++)
        {
            size_t jo = m_pancreas_box[1]+j;
            for(size_t i=0;i<dim[0];i++)
            {
                size_t io = m_pancreas_box[0]+i;
                size_t indx  = i  + j *dim[0] + k *dim[0]*dim[1];
                size_t indxo = io + jo*segdim[0] + ko*segdim[0]*segdim[1];
                ushort val = (*m_seg)[indxo];
                if( val == segLab )
                    (**newVol)[indx] = 1.0f;
            }
        }
    }

    /*
    vpVolAlgoSmooth smooth;
    smooth.SetDataSet(*newVol);
    if( segLab == LABEL_PANCREAS)
        smooth.SetNumberOfPasses(1);
    else
        smooth.SetNumberOfPasses(1);

    int success = smooth.RunAlgo();
    if( !success )
    {
        MacroWarning("Smoothing algorithm failure.");
        return 0;
    }
    */

    return 1;
}

int vpTextureContext::GetPancBoxInTexCoord(float box[6])
{
    MacroConfirmOrReturn(box,0);
    MacroConfirmOrReturn(m_vol,0);
    MacroConfirmOrReturn(m_panc,0);

    size_t voldim[3]={0,0,0};
    m_vol->GetDimensions(voldim);

    box[0] = float(m_pancreas_box[0]) / float(voldim[0]-1);
    box[1] = float(m_pancreas_box[1]) / float(voldim[1]-1);
    box[2] = float(m_pancreas_box[2]) / float(voldim[2]-1);

    box[3] = float(m_pancreas_box[3]) / float(voldim[0]-1);
    box[4] = float(m_pancreas_box[4]) / float(voldim[1]-1);
    box[5] = float(m_pancreas_box[5]) / float(voldim[2]-1);

    return 1;
}
