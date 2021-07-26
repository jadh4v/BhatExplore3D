#pragma once
#include <atomic>
#include <future>

#include <vtkNew.h>
#include <vtkNamedColors.h>
#include <vtkImageData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkBoxWidget.h>
#include <vtkImagePlaneWidget.h>
#include <vtkBorderWidget.h>
#include <vtkActor.h>
#include <QVTKOpenGLNativeWidget.h>
#include <QMainWindow>
#include <QElapsedTimer>
//#include <QtQuickWidgets/QQuickWidget>
//#include <QVTKRenderWidget.h>
#include <vtkContextView.h>
#include <vtkImageSlice.h>
#include <vtkImplicitFunction.h>
//#include <QVTKWidget.h>
#include "MultiBlockSync.h"
#include "TreeView.h"
#include "HistogramViewer.h"
#include "GenericExplorer/Bookmark.h"
#include "ds/Image.h"
#include "VolumeVisualizer.h"

#define HIERARCHICAL
#define QUICK_SELECT
#define DIM 1
#if (DIM == 1)
    #define ATTRIB float
#elif(DIM == 2)
    #define ATTRIB float2
#elif(DIM == 3)
    #define ATTRIB float3
#endif

template<size_t _Dim, typename _Attrib> class MultiBlockSync;
template<size_t _Dim, typename _Attrib> class ActiveContourBlock;
typedef ActiveContourBlock<DIM, ATTRIB> BlockType;
typedef MultiBlockSync<DIM, ATTRIB> BlockSyncType;

using QVTKRenderWidget = QVTKOpenGLNativeWidget;

class QPushButton;
class BookmarkTool;
class DialogOpticalProperties;

class UserInterface : public QMainWindow
{
Q_OBJECT
public:
    typedef vtkSmartPointer<vtkImageData> ImagePtr;
    typedef vtkSmartPointer<vtkImplicitFunction> ImplicitFunction;
    UserInterface(ImagePtr inputImage, ImagePtr displayImage);
    virtual ~UserInterface();
    void UpdateVisualizer();
    MacroSetMember(int, m_NumOfBlocks, NumberOfBlocks)

private:
    // Private functions
    bool _Is2DMode() const;
    bool _Is3DMode() const;
    void _Dock();
    void _PrepareRenderer();
    void _ROI();
    void _OpticalDialog();
    void _Construct_3D_UI();
    void _Construct_2D_UI();
    void _Process2DSelection(ImagePtr& data, ImagePtr& mask);
    void _Process3DSelection(ImagePtr& data, ImagePtr& mask, ImplicitFunction& func);
    QVTKInteractor* _Interactor() const;
    vtkRenderWindow* _Window() const;
    void _SaveCurrentPhi();
    void _RegionToBookmark(ImagePtr region);
    void _SaveBookmarks();
    struct Mask;
    void ComputeBookmarkRegion(Mask& mask, ImagePtr data);

    vtkSmartPointer<vtkPlanes> _GetImplicitSelectionRegion(double maskBounds[6]);
    void _ConstructPImage(ImagePtr& P);
    void _RemoveStructureFromVolume(ImagePtr structureMask, ImagePtr volume, bool invertMask);

    // Member variables
    int m_NumOfBlocks = 1;
    ImagePtr m_InputImage = nullptr;
    sjDS::Image m_VisImage;
    ImagePtr m_Input2D = nullptr;
    struct {
    public:
        vtkSmartPointer<vtkImageData> data;
        vtkSmartPointer<vtkPolyData> contour;
        vtkSmartPointer<vtkPolyDataMapper> mapper;
        vtkSmartPointer<vtkActor> actor;
        vtkSmartPointer<vtkContourFilter> contourFilter;
        void Initialize(vtkNamedColors* colors)
        {
            contour = vtkSmartPointer<vtkPolyData>::New();
            mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            actor = vtkSmartPointer<vtkActor>::New();
            contourFilter = vtkSmartPointer<vtkContourFilter>::New();

            mapper->ScalarVisibilityOff();
            actor->SetMapper(mapper);
            actor->GetProperty()->SetColor(colors->GetColor3d("Red").GetData());
            actor->GetProperty()->SetOpacity(0.3);
            actor->GetProperty()->SetLineWidth(2.0);
        }
    } m_Phi;
    ImagePtr m_Mask = nullptr;
    ImagePtr m_PinImage = nullptr;
    ImagePtr m_PoutImage = nullptr;
    std::future<vtkSmartPointer<vtkRenderWindow>> m_WindowFuture;
    std::atomic<bool> m_Stop = false;
    vtkNew<vtkBorderWidget> m_Border;
    vtkNew<vtkBoxWidget> m_Box;
    vtkNew<vtkImagePlaneWidget> m_Plane;
    vtkNew<vtkActor> m_OutlineActor;
    vtkNew<vtkNamedColors> m_Colors;
    std::vector<std::future<void>> m_Futures;
    BlockSyncType* m_BlockManager = nullptr;
    QElapsedTimer m_Timer;

    // UI variables
    struct {
        vtkNew<vtkRenderer> Ren;
        QVTKRenderWidget* RenWidget = nullptr;
        QPushButton* GenMaskButton = 0;
        HistogramViewer* wHistogramViewer = nullptr;
        TreeView* wTreeView = nullptr;
        VolumeVisualizer* wVolumeVisualizer = nullptr;
        BookmarkTool* wBookmarkTool = nullptr;
        QWidget* wROI = nullptr;
        DialogOpticalProperties* wTFEditor = nullptr;
    }m_ui;

    const bool m_OverrideEnabled = true;
    const double m_SelectionExpansionFactor = 2;
    const unsigned long m_RefreshTime = 2000;
    std::vector<float> m_Samples;
    std::vector<float> m_Pin, m_Pout;

    struct Region{
        //std::vector<vtkSmartPointer<vtkActor>> actors;
        //std::vector<ImagePtr> nodes;
        //std::vector<Bookmark> bookmarks;
        vtkSmartPointer<vtkActor> actor;
        ImagePtr node;
        Bookmark bookmark;
        Region() { actor = vtkSmartPointer<vtkActor>::New(); node = vtkSmartPointer<vtkImageData>::New(); };
    };

    std::vector<Region> m_structures;

    struct Mask{
        ImagePtr buffer;
        void Reset();
        void MapPartialMask(ImagePtr partial);
        void Dilate(int size);
        void Invert();
    }m_FullMask;

private slots:
    void slot_GenMask();
    void slot_StopACBlock();
    void slot_CancelACBlock();
    void slot_SampleBackground();
    void slot_PlaneOrientation();
    void slot_SaveRegions();
    void slot_ShowPlane(bool);
    void slot_ShowOutline(bool);
    void slot_UpdateOpacity(int);
    void slot_UpdateColorProperty(size_t,int,int,int,int);
    void slot_SaveROI();
    void slot_LoadROI();
    void slot_NodeChanged(size_t);
    void slot_showHistogramViewer(bool);
    void slot_ClippingChanged();
    void slot_AddBookmark();
};

