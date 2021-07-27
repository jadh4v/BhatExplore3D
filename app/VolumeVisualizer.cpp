#ifdef _WIN32
    #define NOMINMAX
    #include <Windows.h>
    #undef NOMINMAX
#endif

#include <vector>
#include <queue>
#include <list>
#include <set>
#include <map>
#include <fstream>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

#include <QFileDialog>
#include <QTreeWidget>
#include <QListWidget>
#include <QBoxLayout>
#include <QPushButton>
#include <QTableWidget>
#include <QHeaderView>
#include <QGroupBox>
//#include <QWebEngineView>
//#include <QWebEnginePage>
#include <QResizeEvent>
#include <QSpacerItem>
#include <QMessageBox>
#include <QMenuBar>
#include <QLabel>

#include <vtkGraphLayoutView.h>
#include <vtkRenderedGraphRepresentation.h>
#include <vtkTextProperty.h>
#include <vtkPlot.h>
#include <vtkChartXY.h>
#include <vtkFloatArray.h>
#include <vtkTable.h>
//#include <QVTKWidget.h>
#include <vtkContextView.h>
#include <vtkContextScene.h>
#include <vtkContextItem.h>
#include <vtkDendrogramItem.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkMutableDirectedGraph.h>
#include <vtkUnsignedIntArray.h>
#include <vtkStringArray.h>
#include <vtkDataSetAttributes.h>
#include <vtkMetaImageWriter.h>

#include "AlgoImageToVtk.h"
#include "AlgoVtkToImage.h"
#include "vpStructures.h"
#include "graphics/TextureContext.h"
#include "graphics/VolumeRenderer.h"
#include "graphics/AxialView.h"
#include "ds/Grid.h"
#include "ds/Image.h"
#include "ImageSegmentation.h"
#include "ds/VoxelRow.h"
#include "CompressedSegmentation.h"
#include "ds/GridPoint.h"
#include "ds/BitImage.h"
#include "io/VolumeReader.h"
#include "../GenericExplorer/BookmarkTool.h"
#include "VolumeVisualizer.h"
#include "Slider.h"

using std::cout;
using std::endl;
using std::vector;
using std::list;
using std::set;
using std::map;
using std::pair;
using std::make_pair;
using sjDS::Grid;
using sjDS::GridPoint;
using sjDS::Image;
using sjDS::VoxelRow;
using sjDS::CompressedSegmentation;

extern std::fstream gLog;//("./log.txt", std::fstream::out );

extern Grid globalHistGrid;
VolumeVisualizer::VolumeVisualizer(QWidget* parent) : QWidget(parent)
{
    m_Context = new TextureContext(this);

#ifdef AXIAL_VIEW
    m_AxialView = new AxialView(this);
    m_AxialSlider = new QSlider(Qt::Horizontal);
    m_AxialSlider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_AxialSlider->setFixedHeight(16);
    m_AxialSlider->setDisabled(true);
    m_AxialView->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    m_AxialView->setFixedSize( 512, 512 );
    m_AxialView->setVisible(false);
    m_AxialSlider->setVisible(false);
    // 2D view part
    QVBoxLayout* SliceAndSlider = new QVBoxLayout();
    SliceAndSlider->addWidget( m_AxialView );
    SliceAndSlider->addWidget( m_AxialSlider );
#endif

    m_VolRen = new VolumeRenderer(this);
    m_VolRen->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_VolRen->setMinimumSize(512, 512);
    m_VolRen->SetRayProgramPath("F:/github/BhatExplore3D/Graphics/volume_rendering.vert.glsl", "F:/github/BhatExplore3D/app/volume_rendering.frag.glsl");
    QHBoxLayout* layout = new QHBoxLayout();
    this->setLayout(layout);
    layout->addWidget(m_VolRen);
#ifdef AXIAL_VIEW
    //layout->addLayout(SliceAndSlider);
    QObject::connect( m_AxialSlider,   SIGNAL(valueChanged(int)), m_AxialView, SLOT(slot_updateSlice(int)) );
#endif
    /*
    auto tools = this->menuBar()->addMenu("&Tools");
    auto dice = new QAction(tr("&Dice"), this);
    tools->addAction(dice);
    QObject::connect(dice, SIGNAL(triggered(bool)), this, SLOT(slot_dice(bool)));
    */

}

VolumeVisualizer::~VolumeVisualizer()
{
    MacroDelete(m_AxialView);
    MacroDelete(m_Context);
    MacroDelete(m_RenderedSegmentation);
    MacroDelete(m_Histogram);
}

void VolumeVisualizer::SetImage(vtkSmartPointer<vtkImageData> mainVolume)
{
    m_OriginalImage = mainVolume;
    m_Context->makeCurrent();
    // Construct texture context for different rendering modes, which will share these common volume, segmentation, and optical property textures.
    sjDS::Grid g(m_OriginalImage);
    m_RenderedSegmentation = new Image(&g);
    //m_Context->SetImage(&mainVolume);
    m_Context->SetSeg(m_RenderedSegmentation);
    size_t opt_dim[3] = { TextureContext::sOpticalTextureResolution, TextureContext::sOpticalTextureResolution, 1 };
    m_Optical = Image( opt_dim );
    m_Context->SetOpticalProperties( &m_Optical );

    m_Context->Refresh();

    //TextureHandle VolTex        = m_Context->GetVolumeTexture();
    //TextureHandle VolTexArray   = m_Context->GetVolumeTextureArray();
    if(m_OriginalImage->GetNumberOfScalarComponents() == 1)
        _CreateVolumeTexture_GrayScale();
    else
        _CreateVolumeTexture_RGBA();

    TextureHandle SegTex        = m_Context->GetSegmentTexture();
    TextureHandle SegTexArray   = m_Context->GetSegmentTextureArray();
    TextureHandle hdl_optical   = m_Context->GetOpticalTexture();

    double range[2];
    m_OriginalImage->GetScalarRange(range);
    //int max_value = range[1] - range[0];
    int max_value = (int)range[1];
    cout << "m_axial maxvalue = " << max_value << endl;
    //float color_window[2] = { max_value/2.0f, max_value/2.0f};
    //float color_window[2] = { -90, 410 };
    float color_window[2] = { -90, 1000 };
    //float color_window[2] = { 32000, 50000 };

#ifdef AXIAL_VEW
    m_AxialView->SetVolumeTextureArray( VolTexArray.opengl_tex );
    m_AxialView->SetSegmentTexture(SegTex.opengl_tex);
    m_AxialView->SetSegmentTextureArray(SegTexArray.opengl_tex);
    m_AxialView->SetOpticalTexture( hdl_optical.opengl_tex );
    m_AxialView->SetVolumeTextureMaxValue( max_value );
    m_AxialView->SetDicomColorWindow( color_window );
    m_AxialView->SetGrid(mainVolume.GetGrid());
    m_AxialView->SetSpacing( mainVolume.GetGrid()->Spacing() );
#endif

    /*
    m_AxialSlider->setEnabled(true);
    m_AxialSlider->setRange( 0, (int) mainVolume.GetGrid()->z() );
    */
    // Compute Display Bounding Box
    //const Grid* grid = mainVolume.GetGrid();
    size_t dim[3], corners[6];
    g.GetDimensions(dim);
    corners[0] = corners[1] = corners[2] = 0;
    corners[3] = dim[0]-1;
    corners[4] = dim[1]-1;
    corners[5] = dim[2]-1;

    // Set Volume and Display Parameters
    m_VolRen->SetGrid(&g);
    m_VolRen->SetVoxelSpacing( g.Spacing() );
    m_VolRen->SetVolumeDimensions( dim );
    m_VolRen->SetDisplayBoundingBox( corners );

    // Set Volume Texture
    m_VolRen->SetVolumeTexture(m_VolTex.opengl_tex);
    //m_VolRen->SetVolumeTextureArray( VolTexArray.opengl_tex );
    m_VolRen->SetVolumeTextureScaleOffset(max_value, 0);

    // Set Segmentation Texture
    m_VolRen->SetSegmentTexture( SegTex.opengl_tex );
    m_VolRen->SetSegmentTextureArray(SegTexArray.opengl_tex);
    m_VolRen->SetOpticalTexture( hdl_optical.opengl_tex );
}

void VolumeVisualizer::_ConstructOpticalProperties(size_t numOfSegments)
{
    m_Optical.ClearData();
    size_t maxSize = (size_t)TextureContext::sOpticalTextureResolution;
    size_t minSize = 1;
    numOfSegments = vtkMath::ClampValue( numOfSegments, minSize, maxSize );

    struct myRgba{
    public:
        uchar r : 8, g : 8, b : 8, a : 8;
        myRgba() : r(0), g(0), b(0), a(0) {}
        uint ToUInt() const
        {
            uint ret = (uint(r) << 24) + (uint(g) << 16) + (uint(b) << 8) + uint(a);
            return ret;
        }
    };

    myRgba color;
    color.r = 255;

    // set first row to all zeroes:
    for(size_t i=0; i < maxSize; ++i)
        m_Optical.SetVoxel(i, 0, 0, 0);

    float offset = 10.0f;
    float stepsize = (255.0f-offset) / float(numOfSegments-1);
    for( size_t i=1; i < numOfSegments; ++i)
    {
        float alpha = offset + float(stepsize*float(i));
        //hseg::clamp( alpha, 0.0f, 255.0f);
        alpha = vtkMath::ClampValue(alpha, 0.0f, 255.0f);
        color.a = (uchar)255;
        color.r = (uchar)alpha;
        color.b = 255-(uchar)alpha;
        color.g = (uchar)100;

        for(size_t p=0; p < maxSize; ++p)
            m_Optical.SetVoxel( p, i, 0, color.ToUInt());
    }

    m_Context->UpdateOpticalTexture();
}

void VolumeVisualizer::_UpdateRenderedSegmentation(const set<size_t>& voxels, uint value)
{
    for( auto v : voxels )
        m_RenderedSegmentation->SetVoxel((type_uint)v, value);
}

void VolumeVisualizer::_UpdateRenderedSegmentation(const vector<VoxelRow>& voxRows, uint value)
{
    for( auto v : voxRows )
        m_RenderedSegmentation->SetVoxelRow(v, value);
}

void VolumeVisualizer::_UpdateRenderedSegmentationAccumulative(const vector<uint>& superVoxels, uint& highestLabelCount)
{
    for( uint sv : superVoxels )
    {
        vector<VoxelRow> voxRows;
        m_Seg.GetRegion( sv, voxRows );
        _UpdateRenderedSegmentationAccumulative( voxRows, highestLabelCount );
    }
}

void VolumeVisualizer::_UpdateRenderedSegmentationAccumulative(const vector<VoxelRow>& voxRows, uint& highestLabelCount)
{
    uint* ptr = m_RenderedSegmentation->GetDataPointer();
    for( const VoxelRow& row : voxRows )
    {
        for(auto v = row.Start(); !row.atEnd(v); ++v )
        {
            uint prev = ptr[v];
            ptr[v] = prev+1;
            //uint prev = m_RenderedSegmentation->GetVoxel(v);
            //m_RenderedSegmentation->SetVoxel(v, prev+1);
            if( prev+1 > highestLabelCount)
                highestLabelCount = prev+1;
        }
    }
}

void VolumeVisualizer::slot_UpdateRendering(const std::vector<Bookmark*>& bookmarks)
{
    m_RenderedSegmentation->ClearData();

    uint segNumber = 1;
    //vector<QColor> segColors;
    vector<vector<uint>> segTFs;
    vector<pair<uint,uint>> segRanges;

    //segColors.reserve( bookmarks.size() );
    m_VolRen->SetNumberOfSegments( bookmarks.size()+1 );

    for( const Bookmark* b : bookmarks )
    {
        if( b == nullptr || !b->IsChecked() )
            continue;

        const vector<VoxelRow>& r = b->GetRegion();
        vpTransferFunction tf = b->GetTF();
        vector<uint> samples((size_t)TextureContext::sOpticalTextureResolution);
        auto ref_range = tf.GetReferenceRange();
        double adjustedRange[2] = { ref_range.first, ref_range.second };
        tf.SampleFunction( TextureContext::sOpticalTextureResolution-1, adjustedRange, &samples[0] );
        segTFs.push_back( samples );

        if( b->GetMode() )
            m_VolRen->SetRenderMode( segNumber, VolumeRenderer::RENMODE_SURFACE );
        else
            m_VolRen->SetRenderMode( segNumber, VolumeRenderer::RENMODE_DEFAULT );

        _UpdateRenderedSegmentation(r, segNumber);
        uint range[2] = {UINT_MAX, 0};
        _GetScalarRange(range, r);
        m_VolRen->SetSegmentRange( segNumber, range );
        //segRanges.push_back( make_pair(range[0], range[1]) );
        segNumber++;
    }

    _ConstructOpticalProperties( segTFs, segRanges );
    m_Context->UpdateSegTexture();
#ifdef AXIAL_VEW
    m_AxialView->update();
#endif
    m_VolRen->SetModeToConsiderScalars();
    m_VolRen->update();
}

void VolumeVisualizer::_ConstructOpticalProperties(const vector<vector<uint>>& segTFs, const vector<pair<uint,uint>>& segRanges)
{
    size_t maxSize = TextureContext::sOpticalTextureResolution;
    uint* ptr = m_Optical.GetDataPointer();
    int tf_index = 1;
    for(auto& tf : segTFs)
    {
        if( tf_index >= maxSize-1 )
            break;

        memcpy(&ptr[tf_index*TextureContext::sOpticalTextureResolution], &tf[0], sizeof(uint)*TextureContext::sOpticalTextureResolution);
        ++tf_index;
    }

    //assert( maxSize > 2);
    //for(size_t i=0; i < segRanges.size(); ++i)
    //{
    //    ptr[maxSize*(maxSize-2) + i] = segRanges[i].first;
    //    ptr[maxSize*(maxSize-1) + i] = segRanges[i].second;
    //}

    m_Context->UpdateOpticalTexture();
}

void VolumeVisualizer::slot_saveBookmarks(const std::vector<Bookmark*>& all_bookmarks)
{
    if( all_bookmarks.empty() )
        return;

    QString fileName = QFileDialog::getSaveFileName( this, "Save Bookmarks to file.", QDir::currentPath(), "*.bmk" );
    QFile saveFile(fileName);
    if( !saveFile.open(QFile::WriteOnly) )
    {
        MacroWarning("Cannot open save file.");
        return;
    }

    QDataStream stream( &saveFile );
    stream << (quint64) all_bookmarks.size();

    for(auto b : all_bookmarks)
        b->Write(stream);

    saveFile.flush();
    saveFile.close();
}

void VolumeVisualizer::slot_loadBookmarks(BookmarkTool* bookmarkTool)
{
    QString fileName = QFileDialog::getOpenFileName( this, "Load Bookmarks from file.", QDir::currentPath(), "*.bmk" );
    QFile loadFile(fileName);
    if( !loadFile.open(QFile::ReadOnly) )
    {
        MacroWarning("Cannot open load file.");
        return;
    }

    QDataStream stream( &loadFile );

    quint64 bookmarkCount = 0;
    stream >> bookmarkCount;

    if( bookmarkCount == 0 )
     {
        loadFile.close();
        return;
    }

    int ret = QMessageBox::warning((QWidget*)this, QString("Merge"), QString("Merge with existing Bookmarks?"), QMessageBox::Yes, QMessageBox::No);
    if(ret == QMessageBox::No)
        bookmarkTool->Clear();

    for( quint64 i=0; i < bookmarkCount; ++i )
    {
        Bookmark* b = Bookmark::Read( stream );
        if( !b )
        {
            MacroWarning("Unable to continue loading bookmarks file.");
            return;
        }

        bookmarkTool->AddBookmark(b);
    }

    // Update the opengl texture representing optical properties:
}

#include <vtkImageResize.h>
void VolumeVisualizer::slot_dice(bool t)
{
    QString filename = QFileDialog::getOpenFileName(0, "Open File", "D:/ImagingData/");
    if (filename.isEmpty())
        filename = QFileDialog::getExistingDirectory(0, "Open DIR", "D:/ImagingData/");
    VolumeReader reader(filename);
    double diceOverlap = 0;
    double a_and_b = 0, a_or_b = 1;
    if (reader.Read())
    {
        auto vtk_mask = reader.GetVtkOutput();
        int dim[3];
        vtk_mask->GetDimensions(dim);
        vtkNew<vtkImageResize> resize;
        resize->SetInputData(vtk_mask);
        int f = 2;
        resize->SetOutputDimensions(dim[0] / f, dim[1] / f, dim[2] / f);
        resize->Update();
        vtk_mask = resize->GetOutput();
        auto mask = hseg::AlgoVtkToImage::Convert(vtk_mask);
        //sjDS::Image mask = reader.GetOutput();
        for (size_t i = 0; i < mask.GetArraySize(); ++i)
        {
            if (mask[i] != 0 && (*m_RenderedSegmentation)[i] != 0)
                a_and_b += 1;
            if(mask[i] != 0 || (*m_RenderedSegmentation)[i] != 0)
                a_or_b += 1;
        }
        for (size_t i = 0; i < mask.GetArraySize(); ++i)
        {
            if (mask[i] != 0)
                (*m_RenderedSegmentation)[i] = 1;
            else
                (*m_RenderedSegmentation)[i] = 0;
        }
    }

    //m_RenderedSegmentation->Write("dice_seg.mhd");
    auto seg = hseg::AlgoImageToVtk::Convert(*m_RenderedSegmentation, VTK_CHAR);
    vtkNew<vtkMetaImageWriter> writer;
    writer->SetInputData(seg);
    writer->SetFileName("dice_seg.mhd");
    writer->SetCompression(0);
    writer->Write();

    diceOverlap = a_and_b / a_or_b;
    MacroPrint(a_and_b);
    MacroPrint(a_or_b);
    MacroPrint(diceOverlap);
    m_Context->UpdateSegTexture();
#ifdef AXIAL_VEW
    m_AxialView->update();
#endif
    m_VolRen->SetModeToConsiderScalars();
    m_VolRen->update();
}

void VolumeVisualizer::ClipVolume(float value[6])
{
    int dim[3];
    m_OriginalImage->GetDimensions(dim);
    size_t corners[6];
    m_VolRen->GetDisplayBoundingBox(corners);
    for (int side = 0; side < 6; ++side)
    {
        //corners[side] = (size_t) double(dim[side % 3])*double(value) / 100.0;
        corners[side] = (size_t) float(dim[side % 3]) * value[side];
    }
    m_VolRen->SetDisplayBoundingBox( corners );
    m_VolRen->update();
}

void VolumeVisualizer::showEvent(QShowEvent* event)
{
    MacroMessage("Show event.");
}

void VolumeVisualizer::_CreateVolumeTexture_GrayScale()
{
    MacroConfirm(m_OriginalImage->GetScalarType() == VTK_UNSIGNED_SHORT);
    m_Context->makeCurrent();
    MacroglErrorCheck();

    int dim[3];
    m_OriginalImage->GetDimensions(dim);
    const ushort* volumeData = static_cast<const ushort*>(m_OriginalImage->GetScalarPointer());
    size_t arraySize = size_t(dim[0] * dim[1] * dim[2]);
    vector<float> conv_buffer(arraySize);
    std::transform(volumeData, volumeData + arraySize, conv_buffer.data(), [](const ushort value) { return float(value); });

    if (m_VolTex.opengl_tex == 0)
    {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glGenTextures(1, &m_VolTex.opengl_tex);
        glBindTexture(GL_TEXTURE_3D, m_VolTex.opengl_tex);
        //glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED, GL_FLOAT, conv_buffer.data());
            //(GLsizei)dim[2], 0, GL_RED, GL_UNSIGNED_SHORT, volumeData);
    }
    else
    {
        glBindTexture(GL_TEXTURE_3D, m_VolTex.opengl_tex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RED, GL_FLOAT, conv_buffer.data());
            //(GLsizei)dim[2], 0, GL_RED, GL_UNSIGNED_SHORT, volumeData);
    }
    MacroglErrorCheck();
    m_Context->doneCurrent();
}

void VolumeVisualizer::_CreateVolumeTexture_RGBA()
{
    MacroConfirm(m_OriginalImage->GetScalarType() == VTK_UNSIGNED_CHAR);
    int comp = m_OriginalImage->GetNumberOfScalarComponents();
    MacroConfirm(comp >= 2 && comp <= 3);
    std::vector<uchar> data_rgb;
    data_rgb.resize(m_OriginalImage->GetNumberOfPoints() * 3);
    const uchar* vtk_ptr = (const uchar*)m_OriginalImage->GetScalarPointer();
    for (vtkIdType i = 0; i < m_OriginalImage->GetNumberOfPoints(); ++i)
    {
        for (vtkIdType j = 0; j < 3; ++j)
        {
            if (j < comp)
                data_rgb[i*3 + j] = vtk_ptr[i*comp + j];
            else
                data_rgb[i*3 + j] = uchar(0);
        }
    }

    m_Context->makeCurrent();
    MacroglErrorCheck();

    int dim[3];
    m_OriginalImage->GetDimensions(dim);
    const void* volumeData = m_OriginalImage->GetScalarPointer();

    if (m_VolTex.opengl_tex == 0)
    {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glGenTextures(1, &m_VolTex.opengl_tex);
        glBindTexture(GL_TEXTURE_3D, m_VolTex.opengl_tex);
        //glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RGB, GL_UNSIGNED_BYTE, (const void*)data_rgb.data());
    }
    else
    {
        glBindTexture(GL_TEXTURE_3D, m_VolTex.opengl_tex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB, (GLsizei)dim[0], (GLsizei)dim[1],
            (GLsizei)dim[2], 0, GL_RGB, GL_UNSIGNED_BYTE, (const void*)data_rgb.data());
    }
    MacroglErrorCheck();
    m_Context->doneCurrent();
}

int VolumeVisualizer::_GetScalarRange(type_uint range[2], const std::vector<sjDS::VoxelRow>& region) const
{
    int ret = 1;
    range[0] = UINT_MAX; range[1] = 0;

    if (m_OriginalImage->GetScalarType() == VTK_UNSIGNED_CHAR)
    {
        range[0] = 0;
        range[1] = 255;
        return ret;
    }

    MacroConfirmOrReturn(m_OriginalImage->GetScalarType() == VTK_UNSIGNED_SHORT, 0);
    MacroConfirmOrReturn(m_OriginalImage->GetNumberOfScalarComponents() == 1, 0);
    const ushort* ptr = (const ushort*)m_OriginalImage->GetScalarPointer();
    for( auto& r : region)
    {
        for( sjDS::voxelNo_t v = r.Start(); !r.atEnd(v); ++v )
        {
            uint value = (unsigned int)ptr[v];
            range[0] = value < range[0]? value : range[0];
            range[1] = value > range[1]? value : range[1];
        }
    }

    return ret;
}
