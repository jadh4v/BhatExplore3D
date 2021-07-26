#include <vector>
#include <algorithm>
#include <numeric>
// Qt
#include <QAction>
#include <QLayout>
#include <QDockWidget>
#include <QPushButton>
#include <QToolBar>
#include <QSlider>
#include <QTableWidget>
#include <QFile>
#include <QFileDialog>
#include <QDataStream>
#include <QMenuBar>
#include <QQuickItem>
#include <QQmlProperty>
#include <QElapsedTimer>

// Rendering support classes
#include <QVTKInteractor.h>
#include <vtkActor.h>
#include <vtkBarChartActor.h>
#include <vtkLegendBoxActor.h>
#include <vtkNamedColors.h>
#include <vtkRenderer.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkImageActor.h>
#include <vtkImageSlice.h>
#include <vtkImageSliceMapper.h>
#include <vtkImageMapper3D.h>
#include <vtkInteractorStyleImage.h>
#include <vtkCamera.h>
#include <vtkChartXY.h>
#include <vtkContextScene.h>
#include <vtkPlot.h> 
#include <vtkImageProperty.h> 
#include <vtkScalarsToColors.h>
//#include <vtkOSPRayPass.h>
#include <vtkPNGWriter.h>
#include <vtkImageCast.h>
#include <vtkLookupTable.h>

// Interaction widgets
#include <vtkBoxWidget.h>
#include <vtkImagePlaneWidget.h>
#include <vtkBorderRepresentation.h>

// Processing Filters
#include <vtkOutlineFilter.h>
#include <vtkContourFilter.h>
#include <vtkPlanes.h>
#include <vtkImplicitFunctionToImageStencil.h>
#include <vtkImageStencilToImage.h>
#include <vtkExtractVOI.h>
#include <vtkImageStencil.h>
#include <vtkImageClip.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkImageDilateErode3D.h>

// Other VTK structures
#include <vtkPointData.h>
#include <vtkTable.h>
#include <vtkFloatArray.h>

// App
#include "ROIBoxCallback.h"
#include "RenderCallback.h"
#include "UserInterface.h"
#include "BhattParameters.h"
#include "ActiveContourBlock.h"
#include "ProjUtils.h"
#include "core/macros.h"
#include "utils/utils.h"
#include "../GenericExplorer/Bookmark.h"
#include "../GenericExplorer/BookmarkTool.h"
#include "ds/Image.h"
#include "ds/ImageSegmentation.h"
#include "ds/CompressedSegmentation.h"
#include "DialogOpticalProperties.h"
#include "vpTransferFunctionEditor.h"
#include "ui_roi_clipping_box.h"

typedef unsigned int uint;
typedef vtkSmartPointer<vtkImageData> ImagePtr;
extern BhattParameters gParam;

UserInterface::UserInterface(ImagePtr inputImage, ImagePtr displayImage)
    : QMainWindow()
{
    this->setWindowTitle("Active Contours Volume Exploration");
    ProjUtils::Print::Dimensions(inputImage, "inputImage");
    ProjUtils::Print::Range(inputImage, "inputImage");
    ProjUtils::Print::ComponentCount(inputImage, "inputImage");
    m_InputImage = inputImage;
    m_Input2D = displayImage;
    m_FullMask.buffer = Utils::ConstructImage(m_InputImage, VTK_CHAR);
    m_Phi.Initialize(m_Colors);
    m_Phi.data = vtkSmartPointer<vtkImageData>::New();
    _ConstructPImage(m_PinImage);
    _ConstructPImage(m_PoutImage);
    m_PoutImage->SetOrigin(gParam.HistSize()+100, 0, 0);

    m_ui.RenWidget = new QVTKRenderWidget();
    m_ui.RenWidget->resize(512, 512);
    m_ui.RenWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    //m_ui.RenWidget->setFixedSize(512, 512);

    // Setup render window
    {
        vtkNew<vtkGenericOpenGLRenderWindow> window;
        window->AddRenderer(m_ui.Ren);
        window->SetPosition(900, 0);
        window->SetMultiSamples(8);
        window->SetSize(1000, 1000);
        m_ui.RenWidget->setRenderWindow(window);
        //vtkNew<vtkOSPRayPass> pass;
        //m_ui.Ren->SetPass(pass);
        window->PolygonSmoothingOn();
        window->PointSmoothingOn();
        window->LineSmoothingOn();
        window->SetMultiSamples(8);
    }

    /*
    auto surface = QVTKRenderWidget::defaultFormat();
    surface.setSamples(8);
    m_ui.RenWidget->setFormat(surface);
    m_ui.RenWidget->setEnableHiDPI(true);
    */

    _ROI();

    _OpticalDialog();

    QHBoxLayout* roi_tf = new QHBoxLayout();
    roi_tf->addWidget(m_ui.wROI);
    roi_tf->addWidget(m_ui.wTFEditor);

    QVBoxLayout* bhat_roi_tf = new QVBoxLayout();
    bhat_roi_tf->addWidget(m_ui.RenWidget);
    bhat_roi_tf->addLayout(roi_tf);

    QWidget* central = new QWidget();
    QHBoxLayout* centralLayout = new QHBoxLayout();
    central->setLayout(centralLayout);
    centralLayout->addLayout(bhat_roi_tf);
    m_ui.wVolumeVisualizer = new VolumeVisualizer(this);
    m_ui.wVolumeVisualizer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    centralLayout->addWidget(m_ui.wVolumeVisualizer);
    this->setCentralWidget(central);


    _Dock();

    m_ui.wBookmarkTool->SetTFEditorWidget(m_ui.wTFEditor);

    _PrepareRenderer();
}

void UserInterface::_OpticalDialog()
{
    m_ui.wTFEditor = new DialogOpticalProperties(vpTransferFunction());
    m_ui.wTFEditor->setFixedSize(720, 240);
}

void UserInterface::_ROI()
{
    m_ui.wROI = new QWidget();
    Ui::ROIClippingBox ui_roi;
    ui_roi.setupUi(m_ui.wROI);

    // X-Axis
    QQuickWidget* x_axis = m_ui.wROI->findChild<QQuickWidget*>(QString("x_axis"));
    auto obj = x_axis->rootObject();
    auto list = obj->children();
    QObject::connect(list[0], SIGNAL(moved()), this, SLOT(slot_ClippingChanged()));
    QObject::connect(list[1], SIGNAL(moved()), this, SLOT(slot_ClippingChanged()));

    // Y-Axis
    QQuickWidget* y_axis = m_ui.wROI->findChild<QQuickWidget*>(QString("y_axis"));
    obj = y_axis->rootObject();
    list = obj->children();
    QObject::connect(list[0], SIGNAL(moved()), this, SLOT(slot_ClippingChanged()));
    QObject::connect(list[1], SIGNAL(moved()), this, SLOT(slot_ClippingChanged()));

    // Z-Axis
    QQuickWidget* z_axis = m_ui.wROI->findChild<QQuickWidget*>(QString("z_axis"));
    obj = z_axis->rootObject();
    list = obj->children();
    QObject::connect(list[0], SIGNAL(moved()), this, SLOT(slot_ClippingChanged()));
    QObject::connect(list[1], SIGNAL(moved()), this, SLOT(slot_ClippingChanged()));

}

void UserInterface::_Dock()
{
    //m_ui.GenMaskButton = new QPushButton("Generate Mask");
    QDockWidget *dockWidget = new QDockWidget(tr("Dock Widget"), this);
    dockWidget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    QToolBar* toolbar = new QToolBar("AC execution");
    toolbar->setOrientation(Qt::Orientation::Vertical);
    dockWidget->setWidget(toolbar);
    QAction* gen = new QAction("Generate Mask");
    toolbar->addAction(gen);
    QAction* stop = new QAction("Stop");
    toolbar->addAction(stop);
    QAction* cancel = new QAction("Cancel");
    toolbar->addAction(cancel);
    QAction* sample = new QAction("Sample Background");
    toolbar->addAction(sample);
    QAction* planeOrient = new QAction("Plane Orientation");
    toolbar->addAction(planeOrient);
    QAction* showPlane = new QAction("Show Plane");
    showPlane->setCheckable(true);
    showPlane->setChecked(true);
    toolbar->addAction(showPlane);

    QAction* saveRegions = new QAction("Save Regions");
    toolbar->addAction(saveRegions);

    QAction* saveROI = new QAction("Save ROI");
    toolbar->addAction(saveROI);
    QAction* loadROI = new QAction("Load ROI");
    toolbar->addAction(loadROI);

    QAction* showOutline = new QAction("Show Bounding Box");
    showOutline->setCheckable(true);
    showOutline->setChecked(true);
    toolbar->addAction(showOutline);

    m_ui.wTreeView = new TreeView();
    toolbar->addWidget(m_ui.wTreeView);

    m_ui.wHistogramViewer = new HistogramViewer(m_PinImage, m_PoutImage);
    m_ui.wHistogramViewer->SetGreyScaleValues(6, 15);
    m_ui.wHistogramViewer->setVisible(false);

    m_ui.wBookmarkTool = new BookmarkTool(this);
    m_ui.wBookmarkTool->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    toolbar->addWidget(m_ui.wBookmarkTool);
    m_VisImage = sjDS::Image(m_InputImage);
    double ref_range[2];
    m_InputImage->GetScalarRange(ref_range);
    m_ui.wBookmarkTool->SetRefRange(ref_range[0], ref_range[1]);
    m_ui.wBookmarkTool->SetVolume(&m_VisImage);

    addDockWidget(Qt::LeftDockWidgetArea, dockWidget);

    //this->setMenuBar(new QMenuBar());
    auto toolsMenu = this->menuBar()->addMenu("Tools");
    auto histogramViewerAction = toolsMenu->addAction("Histogram Viewer");
    histogramViewerAction->setCheckable(true);

    // Connect all signals and slots
    //QObject::connect(m_ui.GenMaskButton, SIGNAL(clicked()), this, SLOT(slot_GenMask()));
    QObject::connect(gen, SIGNAL(triggered()), this, SLOT(slot_GenMask()));
    QObject::connect(stop, SIGNAL(triggered()), this, SLOT(slot_StopACBlock()));
    QObject::connect(cancel, SIGNAL(triggered()), this, SLOT(slot_CancelACBlock()));
    QObject::connect(sample, SIGNAL(triggered()), this, SLOT(slot_SampleBackground()));
    QObject::connect(planeOrient, SIGNAL(triggered()), this, SLOT(slot_PlaneOrientation()));
    QObject::connect(showPlane, SIGNAL(triggered(bool)), this, SLOT(slot_ShowPlane(bool)));
    QObject::connect(showOutline, SIGNAL(triggered(bool)), this, SLOT(slot_ShowOutline(bool)));
    QObject::connect(saveRegions, SIGNAL(triggered()), this, SLOT(slot_SaveRegions()));
    QObject::connect(saveROI, SIGNAL(triggered()), this, SLOT(slot_SaveROI()));
    QObject::connect(loadROI, SIGNAL(triggered()), this, SLOT(slot_LoadROI()));
    //QObject::connect(m_ui.wHistogramViewer, SIGNAL(sign_valueChanged(int, int)), this, SLOT(slot_SetHistGreyScale(int, int)));
    QObject::connect(m_ui.wTreeView, SIGNAL(sign_colorChanged(size_t,int,int,int,int)), this, SLOT(slot_UpdateColorProperty(size_t,int,int,int,int)));
    QObject::connect(m_ui.wTreeView, SIGNAL(sign_nodeChanged(size_t)), this, SLOT(slot_NodeChanged(size_t)));
    QObject::connect(histogramViewerAction, SIGNAL(triggered(bool)), this, SLOT(slot_showHistogramViewer(bool)));
    QObject::connect(m_ui.wBookmarkTool, SIGNAL(sign_updateRendering(const std::vector<Bookmark*>&)), m_ui.wVolumeVisualizer, SLOT(slot_UpdateRendering(const std::vector<Bookmark*>&)) );
    QObject::connect(m_ui.wBookmarkTool, SIGNAL(sign_saveMe(const std::vector<Bookmark*>&)), m_ui.wVolumeVisualizer, SLOT(slot_saveBookmarks(const std::vector<Bookmark*>&)) );
    QObject::connect(m_ui.wBookmarkTool, SIGNAL(sign_loadMe(BookmarkTool*)), m_ui.wVolumeVisualizer, SLOT(slot_loadBookmarks(BookmarkTool*)) );
    QObject::connect(m_ui.wBookmarkTool, SIGNAL(sign_bookmarkSelected(const vpTransferFunction&)), m_ui.wTFEditor, SLOT(slot_setTransferFunction(const vpTransferFunction&)));
    QObject::connect(m_ui.wBookmarkTool, SIGNAL(sign_addTriggered()), this, SLOT(slot_AddBookmark()));
    QObject::connect(m_ui.wTFEditor, SIGNAL(sign_TransferFunctionUpdated(const vpTransferFunction*)), m_ui.wBookmarkTool, SLOT(slot_tfupdate(const vpTransferFunction*)));
}

void UserInterface::_ConstructPImage(ImagePtr& P)
{
    int histSize = (int)gParam.HistSize();
    int dim[] = { histSize, histSize, histSize };
    if (DIM == 2) dim[2] = 1;
    if (DIM == 1) dim[1] = 1;

    double spacing[] = { 1,1,1 };
    double origin[] = { 0,0,0 };
    P = Utils::ConstructImage(dim, spacing, origin, VTK_FLOAT);
}


UserInterface::~UserInterface()
{
    //MacroDelete(m_OpenGLWidget);
}

void UserInterface::_PrepareRenderer()
{
    int dim[3];
    m_InputImage->GetDimensions(dim);
    if (dim[2] == 1)        _Construct_2D_UI();
    else if (dim[2] > 1)    _Construct_3D_UI();
    else                    MacroFatalError("Wrong input dimensions. Cannot proceed.");

}

void UserInterface::_Construct_2D_UI()
{
    vtkNew<vtkContourFilter> contourFilter;
    contourFilter->SetInputData(m_Phi.data);
    contourFilter->SetNumberOfContours(1);
    contourFilter->SetValue(0, 0.0);
    //contourFilter->ComputeScalarsOff();
    //contourFilter->ComputeNormalsOn();

    // Create an actor
    vtkNew<vtkImageSlice>  imageActor;
    vtkNew<vtkImageSliceMapper> imageMapper;
    imageMapper->SetInputData(m_Input2D);
    //imageMapper->SetInputData(m_InputImage);
    //imageMapper->SetInputData(m_Phi.data);
    imageActor->SetMapper(imageMapper);
    vtkNew<vtkScalarsToColors> colorMap;
    colorMap->SetVectorModeToRGBColors();
    imageActor->GetProperty()->SetLookupTable(colorMap);

    vtkNew<vtkActor> contourActor;
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(contourFilter->GetOutputPort());
    mapper->ScalarVisibilityOff();
    contourActor->SetMapper(mapper);
    contourActor->GetProperty()->SetColor(m_Colors->GetColor3d("Red").GetData());
    contourActor->GetProperty()->SetLineWidth(2.0);

    vtkNew<vtkInteractorStyleImage> imageInteractor;
    imageInteractor->SetInteractionModeToImage2D();
    m_ui.RenWidget->interactor()->SetInteractorStyle(imageInteractor);
    m_ui.RenWidget->interactor()->Initialize();

    m_Border->SetInteractor(_Interactor());
    m_Border->CreateDefaultRepresentation();
    m_Border->ResizableOn();
    m_Border->SelectableOff();
    m_Border->GetBorderRepresentation()->GetBorderProperty()->SetLineWidth(4.0f);
    m_Border->GetBorderRepresentation()->GetBorderProperty()->SetColor(m_Colors->GetColor3d("Salmon").GetData());
    m_Border->On();

    m_ui.Ren->AddActor(imageActor);
    m_ui.Ren->AddActor(contourActor);
    m_ui.Ren->ResetCamera();
    m_ui.Ren->SetBackground(m_Colors->GetColor3d("White").GetData());

    // Sign up to receive TimerEvent
    vtkNew<RenderCallback> cb;
    cb->SetWindow(_Window());
    //cb->SetActor(contourActor);
    _Interactor()->AddObserver(vtkCommand::TimerEvent, cb);
    int timerId = _Interactor()->CreateRepeatingTimer(m_RefreshTime);
}

void UserInterface::_Construct_3D_UI()
{
    //vtkNew<vtkContourFilter> contourFilter;
    auto contourFilter = m_Phi.contourFilter.Get();
    contourFilter->SetInputData(m_Phi.data);
    contourFilter->SetNumberOfContours(1);
    contourFilter->SetValue(0, 0.0);
    contourFilter->ComputeScalarsOff();
    //contourFilter->ComputeNormalsOn();
    contourFilter->Update();
    //m_Phi.contour->DeepCopy(contourFilter->GetOutput());
    vtkNew<vtkPolyData> tmp;
    tmp->DeepCopy(contourFilter->GetOutput());
    auto contourMapper = m_Phi.mapper.Get();
    contourMapper->SetInputData(tmp);
    contourMapper->ScalarVisibilityOff();

    auto contourActor = m_Phi.actor.Get();
    contourActor->SetMapper(contourMapper);
    contourActor->GetProperty()->SetColor(m_Colors->GetColor3d("Red").GetData());
    contourActor->GetProperty()->SetOpacity(0.3);
    contourActor->GetProperty()->SetLineWidth(2.0);
    m_ui.Ren->AddActor(contourActor);
    //contourActor->VisibilityOff();

    vtkNew<vtkOutlineFilter> outliner;
    outliner->SetInputData(m_InputImage);
    outliner->Update();
    vtkNew<vtkPolyDataMapper> outlineMapper;
    outlineMapper->SetInputData(outliner->GetOutput());
    m_OutlineActor->SetMapper(outlineMapper);
    m_OutlineActor->GetProperty()->SetColor(m_Colors->GetColor3d("Green").GetData());
    m_ui.Ren->AddActor(m_OutlineActor);

    m_ui.Ren->SetBackground(m_Colors->GetColor3d("White").GetData());

    //Add Image cropping widget (roi widget)
    m_Box->SetInteractor(_Interactor());
    m_Box->SetProp3D(m_OutlineActor);
    m_Box->RotationEnabledOff();
    m_Box->GetOutlineProperty()->SetColor(m_Colors->GetColor3d("Green").GetData());
    m_Box->SetHandleSize(0.005);
    m_Box->PlaceWidget();
    vtkNew<ROIBoxCallback> boxCallback;
    m_Box->AddObserver(vtkCommand::InteractionEvent, boxCallback);
    m_Box->On();

    // Add imageplanewidget
    m_Plane->SetInteractor(_Interactor());
    m_Plane->SetInputData(m_InputImage);
    m_Plane->TextureVisibilityOn();
    m_Plane->TextureInterpolateOn();
    m_Plane->UpdatePlacement();
    m_Plane->SetPlaneOrientation(1);
    //m_Plane->SetWindowLevel(168, 162);
    //m_Plane->SetWindowLevel(347, 16);
    //m_Plane->SetWindowLevel(250, 85);
    if (m_InputImage->GetNumberOfScalarComponents() == 1)
    {
        double r[2];
        m_InputImage->GetScalarRange(r);
        m_Plane->SetWindowLevel(r[1] - r[0], (r[1] + r[0]) / 2.0);
    }
    else
    {
        m_Plane->GetLookupTable()->SetVectorModeToRGBColors();
    }

    m_Plane->PlaceWidget();
    m_Plane->On();
    m_Plane->SetSliceIndex(60);

    //m_ui.Ren->GetActiveCamera()->SetPosition(0, -1, 1);
    m_ui.Ren->GetActiveCamera()->SetPosition(0, -1, 0);
    m_ui.Ren->GetActiveCamera()->SetFocalPoint(m_InputImage->GetOrigin());
    m_ui.Ren->GetActiveCamera()->SetViewUp(0, 0, 1);
    m_ui.Ren->ResetCamera();

    // Sign up to receive TimerEvent
    vtkNew<RenderCallback> cb;
    cb->SetWindow(_Window());
    _Interactor()->AddObserver(vtkCommand::TimerEvent, cb);
    int timerId = _Interactor()->CreateRepeatingTimer(m_RefreshTime);

    // Render and start interaction
    //interactor->Start();
}

void UserInterface::slot_GenMask()
{
    MacroMessage("Timer Started.");
    m_Timer.start();
    // Stop any previous block before starting a new one, for now.
    // In future, maybe allow simultaneous blocks execution.
    slot_StopACBlock();

    int dim[3];
    m_InputImage->GetDimensions(dim);

    MacroNewVtkObject(vtkImageData, data);
    MacroNewVtkObject(vtkImageData, mask);

    auto phi = Utils::ConstructImage(m_InputImage->GetDimensions(), m_InputImage->GetSpacing(), m_InputImage->GetOrigin(), VTK_FLOAT);
    m_Phi.data->DeepCopy(phi);
    //m_Phi.mapper->GetInput()->Initialize();
    m_Phi.actor->VisibilityOn();

    ImplicitFunction func;
    if (dim[2] == 1)
        _Process2DSelection(data, mask);
    else
        _Process3DSelection(data, mask, func);

    m_Stop = false;
    MacroDelete(m_BlockManager);
    m_BlockManager = new BlockSyncType(m_NumOfBlocks, m_Phi.data, m_PinImage, m_PoutImage, m_Stop);
    m_Phi.contourFilter->SetInputData(m_Phi.data);
    m_BlockManager->SetPhiContourFilter(m_Phi.contourFilter);
    m_BlockManager->SetPhiContourMapper(m_Phi.mapper);
    m_BlockManager->SetPhiContourActor(m_Phi.actor);
    m_BlockManager->Launch(data, mask, func, m_Futures);
    //auto f = std::async(std::launch::async, &UserInterface::_Launch_AC_Block, this, data, mask, m_Phi.data, m_PinImage, m_PoutImage, func, std::ref(m_Stop));
    //m_Futures.push_back(std::move(f));
}

template<typename T>
void resizeBounds(const T source[6], T out[6])
{
    for (auto i : {0,2,4})
        out[i] = std::max(source[i], out[i]);
    for (auto i : {1,3,5})
        out[i] = std::min(source[i], out[i]);
}

//calculate clipping extents based on clipping bounds
void ClipBoundsToExtents(ImagePtr ref, const double bounds[6], int extents[6])
{
    double inBounds[6];
    std::copy_n(bounds, 6, inBounds);
    double refBounds[6];
    ref->GetBounds(refBounds);
    resizeBounds(refBounds, inBounds);

    int ijk_low[3], ijk_hi[3];
    double pcoords[3];
    double ijk_low_point[3] = { inBounds[0], inBounds[2], inBounds[4] };
    double ijk_hi_point[3] =  { inBounds[1], inBounds[3], inBounds[5] };
    if (ref->GetDimensions()[2] == 1)
    {
        ijk_low_point[2] = ijk_hi_point[2] = 0.0;
    }

    ref->ComputeStructuredCoordinates(ijk_low_point, ijk_low, pcoords);
    ref->ComputeStructuredCoordinates(ijk_hi_point, ijk_hi, pcoords);

    int out[] = { ijk_low[0], ijk_hi[0], ijk_low[1], ijk_hi[1], ijk_low[2], ijk_hi[2] };
    int refExtents[6];
    ref->GetExtent(refExtents);
    resizeBounds(refExtents, out);
    std::copy_n(out, 6, extents);
}

ImagePtr ClipByExtents(const int extents[6], ImagePtr toClip)
{
    int inExt[6];
    std::copy_n(extents, 6, inExt);
    vtkNew<vtkImageClip> clipper;
    clipper->SetInputData(toClip);
    clipper->SetOutputWholeExtent(inExt);
    clipper->ClipDataOn();
    clipper->Update();
    return clipper->GetOutput();
}

bool UserInterface::_Is2DMode() const
{
    if(m_InputImage)
        return (m_InputImage->GetDimensions()[2] == 1);
    else
        return false;
}

bool UserInterface::_Is3DMode() const
{
    if(m_InputImage)
        return (m_InputImage->GetDimensions()[2] > 1);
    else
        return false;
}

vtkSmartPointer<vtkPlanes> UserInterface::_GetImplicitSelectionRegion(double maskBounds[6])
{
    vtkNew<vtkPlanes> planes;
    if (_Is2DMode())
    {
        // Get corner coordinates of the rectangular selection widget in world coordinates.
        auto p1coord = m_Border->GetBorderRepresentation()->GetPositionCoordinate();
        auto p2coord = m_Border->GetBorderRepresentation()->GetPosition2Coordinate();
        const double* p1 = p1coord->GetComputedWorldValue(m_ui.Ren);
        const double* p2 = p2coord->GetComputedWorldValue(m_ui.Ren);

        // calculate mask and base-region bounds
        double bounds[6] = { p1[0], p2[0], p1[1], p2[1], -1.0, 1.0 };
        planes->SetBounds(bounds);
        std::copy_n(bounds, 6, maskBounds);
    }
    else if (_Is3DMode())
    {
        m_Box->GetPlanes(planes);
        vtkNew<vtkPolyData> boxPd;
        m_Box->GetPolyData(boxPd);
        boxPd->GetBounds(maskBounds);
    }
    return planes;
}

void UserInterface::_Process2DSelection(ImagePtr& data, ImagePtr& mask)
{
    // calculate mask and base-region bounds
    double maskBounds[6];
    auto planes = _GetImplicitSelectionRegion(maskBounds);

    double baseBounds[6];
    std::copy_n(maskBounds, 6, baseBounds);
    auto dx = m_SelectionExpansionFactor * (maskBounds[1] - maskBounds[0]);
    auto dy = m_SelectionExpansionFactor * (maskBounds[3] - maskBounds[2]);
    baseBounds[0] -= dx; baseBounds[1] += dx;
    baseBounds[2] -= dy; baseBounds[3] += dy;

    //calculate clipping extents based on clipping bounds
    int baseExtents[6];
    ClipBoundsToExtents(m_InputImage, baseBounds, baseExtents);
    
    // get clipped data to process
    data = ClipByExtents(baseExtents, m_InputImage);

    // calculate full mask and then clip it
    vtkNew<vtkImplicitFunctionToImageStencil> filterCreateStencil;
    filterCreateStencil->SetInput(planes);
    filterCreateStencil->SetInformationInput(m_InputImage);
    vtkNew<vtkImageStencilToImage> filterStencilToImage;
    filterStencilToImage->SetInputConnection(filterCreateStencil->GetOutputPort());
    filterStencilToImage->SetOutputScalarTypeToUnsignedChar();
    filterStencilToImage->Update();
    auto fullMask = filterStencilToImage->GetOutput();

    // clip the full mask
    mask = ClipByExtents(baseExtents, fullMask);
    m_Phi.data->DeepCopy(ClipByExtents(baseExtents, m_Phi.data));
}

void UserInterface::_Process3DSelection(ImagePtr& data, ImagePtr& mask, ImplicitFunction& func)
{
    double maskBounds[6];
    auto planes = _GetImplicitSelectionRegion(maskBounds);
#ifdef QUICK_SELECT
    func = planes;
    //data->DeepCopy(m_InputImage);
    data = m_InputImage;
    return;
#endif

    vtkNew<vtkImplicitFunctionToImageStencil> filterCreateStencil;
    filterCreateStencil->SetInput(planes);
    filterCreateStencil->SetInformationInput(m_InputImage);
    vtkNew<vtkImageStencilToImage> filterStencilToImage;
    filterStencilToImage->SetInputConnection(filterCreateStencil->GetOutputPort());
    filterStencilToImage->SetOutputScalarTypeToUnsignedChar();
    filterStencilToImage->Update();

    // Deep copy, so that pointer address doesn't change.
    // Other objects will need access to this.
    auto fullMask = filterStencilToImage->GetOutput();


    double baseBounds[6];
    std::copy_n(maskBounds, 6, baseBounds);
    auto dx = m_SelectionExpansionFactor * (maskBounds[1] - maskBounds[0]);
    auto dy = m_SelectionExpansionFactor * (maskBounds[3] - maskBounds[2]);
    auto dz = m_SelectionExpansionFactor * (maskBounds[5] - maskBounds[4]);
    baseBounds[0] -= dx; baseBounds[1] += dx;
    baseBounds[2] -= dy; baseBounds[3] += dy;
    baseBounds[4] -= dz; baseBounds[5] += dz;

    //calculate clipping extents based on clipping bounds
    int baseExtents[6];
    ClipBoundsToExtents(m_InputImage, baseBounds, baseExtents);
    
    // get clipped data to process
    data = ClipByExtents(baseExtents, m_InputImage);

    // clip the full mask
    mask = ClipByExtents(baseExtents, fullMask);
    m_Phi.data->DeepCopy(ClipByExtents(baseExtents, m_Phi.data));
}


void UserInterface::slot_StopACBlock()
{
    MacroMessage("(Stopped) Total Time Before SavePhi() = " << m_Timer.elapsed());
    // Set the global termination flag that all other threads will respect.
    m_Stop = true;
    // wait for all threads to finish, and destroy the future object.
    bool activeBlockWasPresent = false;
    while (!m_Futures.empty())
    {
        m_Futures.back().wait();
        m_Futures.pop_back();
        activeBlockWasPresent = true;
    }
    // save the shared output (phi function).
    if(activeBlockWasPresent)
        _SaveCurrentPhi();

    MacroMessage("(Stopped) Total Time elapsed = " << m_Timer.elapsed());
}
void UserInterface::slot_CancelACBlock()
{
    m_Stop = true;
    m_Phi.actor->VisibilityOff();
    while (!m_Futures.empty())
    {
        m_Futures.back().wait();
        //_SaveCurrentPhi();
        m_Futures.pop_back();
    }
    MacroMessage("(Cancelled) Total Time elapsed = " << m_Timer.elapsed());
}

QVTKInteractor* UserInterface::_Interactor() const
{
    return m_ui.RenWidget->interactor();
}

vtkRenderWindow* UserInterface::_Window() const
{
    return m_ui.RenWidget->renderWindow();
}

void UserInterface::_SaveCurrentPhi()
{
    auto contourActor = m_Phi.actor;
    contourActor->GetProperty()->SetColor(m_Colors->GetColor3d("Yellow").GetData());
    m_ui.Ren->AddActor(contourActor);
    m_Phi.Initialize(m_Colors);
    m_ui.Ren->AddActor(m_Phi.actor);
    m_Phi.actor->VisibilityOff();

    // add entry for base image
    if (m_structures.empty())
    {
        m_structures.push_back(UserInterface::Region());
        m_structures.front().node->DeepCopy(m_InputImage);
        m_ui.wTreeView->AddItem(m_structures.size()-1);
    }

    // add entry for current region
    m_structures.push_back(UserInterface::Region());
    m_structures.back().actor = contourActor;
    m_ui.wTreeView->AddItem(m_structures.size()-1);

    _RegionToBookmark(m_Phi.data);

    // clear m_Phi.data
    std::memset(m_Phi.data->GetScalarPointer(), 0, sizeof(float)*m_Phi.data->GetNumberOfPoints());
    m_Phi.data->Modified();
}

void UserInterface::slot_SampleBackground()
{
    MacroNewVtkObject(vtkImageData, data);
    MacroNewVtkObject(vtkImageData, mask);

    double bounds[6];
    auto planes = _GetImplicitSelectionRegion(bounds);

    const float* scalars = (float*)m_InputImage->GetScalarPointer();
    int comp = m_InputImage->GetNumberOfScalarComponents();
    MacroAssert(scalars);
    MacroAssert(m_InputImage->GetScalarType() == VTK_FLOAT);
    for (vtkIdType id = 0; id < m_InputImage->GetNumberOfPoints(); ++id)
    {
        double p[3];
        m_InputImage->GetPoint(id, p);
        if (planes->EvaluateFunction(p) < 0)
        {
            int value = (int)scalars[id*comp];
            if(comp == 3)
            {
                float r = value;
                float g = scalars[id*comp + 1];
                float b = scalars[id*comp + 2];
                value = int((11*r + 16*g + 5*b)/32);
            }
            vtkMath::ClampValue(value, 0, (int)gParam.ScalarRange());
            m_Samples.push_back(value);
        }
    }
    MacroPrint(m_Samples.size());
}

void UserInterface::slot_PlaneOrientation()
{
    int x = this->m_Plane->GetPlaneOrientation();
    this->m_Plane->SetPlaneOrientation((x+1)%3);
}

void UserInterface::slot_ShowPlane(bool checked)
{
    if (checked) m_Plane->On();
    else m_Plane->Off();
}

void UserInterface::slot_ShowOutline(bool checked)
{
    m_OutlineActor->SetVisibility(checked);
}

/*
void UserInterface::slot_SetColorWindowValue(int value)
{
    m_hist.pinActor->GetProperty()->SetColorWindow((double)value);
    m_hist.poutActor->GetProperty()->SetColorWindow((double)value);
    m_hist.Widget->update();
}

void UserInterface::slot_SetColorLevelValue(int value)
{
    m_hist.pinActor->GetProperty()->SetColorLevel((double)value);
    m_hist.poutActor->GetProperty()->SetColorLevel((double)value);
    m_hist.Widget->update();
}

void UserInterface::slot_SetHistGreyScale(int level, int range)
{
    m_hist.pinActor->GetProperty()->SetColorLevel(double(level));
    m_hist.poutActor->GetProperty()->SetColorLevel(double(level));
    m_hist.pinActor->GetProperty()->SetColorWindow(double(range));
    m_hist.poutActor->GetProperty()->SetColorWindow(double(range));
    m_hist.Widget->renderWindow()->Render();
}
*/

void UserInterface::slot_UpdateOpacity(int opacity)
{
    for (auto& s : m_structures)
    {
        s.actor->GetProperty()->SetOpacity(double(opacity) / 255.0);
        s.actor->Modified();
    }
    m_ui.RenWidget->renderWindow()->Render();
}

void UserInterface::slot_UpdateColorProperty(size_t id, int r, int g, int b, int a)
{
    if (id < m_structures.size())
    {
        auto& actor = m_structures[id].actor;
        actor->GetProperty()->SetColor(r / 255.0, g / 255.0, b / 255.0);
        actor->GetProperty()->SetOpacity(a / 255.0);
        m_ui.RenWidget->renderWindow()->Render();
    }
}

void UserInterface::_RegionToBookmark(ImagePtr region)
{
    m_FullMask.MapPartialMask(region);
    Utils::WriteVolume(m_FullMask.buffer, "Saved_Contour.mhd", false);

    vtkNew<vtkImageData> interior;
    vtkNew<vtkImageData> exterior;
    interior->DeepCopy(m_InputImage);
    exterior->DeepCopy(m_InputImage);
    m_FullMask.Dilate(3);
    _RemoveStructureFromVolume(m_FullMask.buffer, interior, true);
    _RemoveStructureFromVolume(m_FullMask.buffer, exterior, false);

    // construct bookmark and related region for interior
    //m_structures.push_back(UserInterface::Region());
    m_structures.back().node = interior;
    ComputeBookmarkRegion(m_FullMask, interior);
    // Invert the mask to handle exterior now.
    m_FullMask.Invert();
    // construct bookmark and related region for exterior
    m_structures.push_back(UserInterface::Region());
    m_structures.back().node = exterior;
    ComputeBookmarkRegion(m_FullMask, exterior);
    m_ui.wTreeView->AddItem(m_structures.size()-1);
}

void UserInterface::ComputeBookmarkRegion(UserInterface::Mask& mask, ImagePtr data)
{
    // Update the mask to remove zero-valued data voxels (as they are blanked to exclude from hierarchical exploration.
    MacroAssert(mask.buffer->GetScalarType() == VTK_CHAR);
    //MacroAssert(data->GetScalarType() == VTK_FLOAT);
    size_t sz = mask.buffer->GetNumberOfPoints();
    char* maskPtr = (char*)mask.buffer->GetScalarPointer();
    ushort* ushort_dataPtr = (ushort*)data->GetScalarPointer();
    uchar* uchar_dataPtr = (uchar*)data->GetScalarPointer();
    const int type = data->GetScalarType();
    NullCheckVoid(maskPtr);
    NullCheckVoid(ushort_dataPtr);
    NullCheckVoid(uchar_dataPtr);
    if(type == VTK_UNSIGNED_SHORT)
        std::transform(maskPtr, maskPtr + sz, ushort_dataPtr, maskPtr, [](const char m, const ushort d) {return d < 1e-5? 0 : m; });
    else
    {
        size_t comp = data->GetNumberOfScalarComponents();
        for (size_t i = 0; i < sz; ++i)
        {
            const char m = maskPtr[i];
            const uchar d = *std::max_element(uchar_dataPtr + i * comp, uchar_dataPtr + (i+1) * comp);
            maskPtr[i] = d < 1 ? 0 : m;
        }
    }

    mask.buffer->Modified();

    // Now construct a compressed region from the mask, so that a bookmark object can be created.
    sjDS::Image image(mask.buffer);
    std::shared_ptr<sjDS::ImageSegmentation> seg;
    seg.reset(image.ConvertToImageSegmentation());
    if (seg != nullptr)
    {
        sjDS::CompressedSegmentation comSeg(seg->GetGridLabels(), seg->GetArraySize(), *seg->GetGrid());
        std::vector<sjDS::VoxelRow> voxRows;
        comSeg.GetRegion(1, voxRows);
        Bookmark b(voxRows, "my_bookmark");
        m_structures.back().bookmark = std::move(b);
    }
}

void UserInterface::_SaveBookmarks()
{
    if (m_structures.empty())
        return;

    QString fileName = QFileDialog::getSaveFileName(this, "Save Bookmarks to file.", QDir::currentPath(), "*.bmk");
    QFile saveFile(fileName);
    MacroOpenQFileToWrite(saveFile);
    QDataStream stream(&saveFile);
    stream << (quint64)m_structures.size();

    for (auto s : m_structures)
        s.bookmark.Write(stream);

    saveFile.flush();
    saveFile.close();
}

void UserInterface::Mask::Reset()
{
    char* buff = (char*)this->buffer->GetScalarPointer();
    size_t sz = (size_t)this->buffer->GetNumberOfPoints();
    std::fill(buff, buff + sz, char(0));
    this->buffer->Modified();
}

void UserInterface::Mask::MapPartialMask(ImagePtr partial)
{
    this->Reset();
    char* buff = (char*)this->buffer->GetScalarPointer();
    float* partial_ptr = (float*)partial->GetScalarPointer();
    vtkIdType partialCount = partial->GetNumberOfPoints();
    for (vtkIdType i = 0; i < partialCount; ++i)
    {
        //if (partial_ptr[i] >= 0)
        //    continue;

        if (partial_ptr[i] < 0)
        {
            double x[3] = { 0,0,0 };
            partial->GetPoint(i, x);
            vtkIdType id = this->buffer->FindPoint(x);
            if (id >= 0)
                buff[id] = char(1);
        }
    }
    this->buffer->Modified();
}

void UserInterface::Mask::Dilate(int size)
{
    vtkNew<vtkImageDilateErode3D> dilate;
    dilate->SetInputData(this->buffer);
    dilate->SetDilateValue(1.0);
    dilate->SetErodeValue(0.0);
    dilate->SetNumberOfThreads(16);
    dilate->SetKernelSize(size, size, size);
    dilate->Update();
    this->buffer = dilate->GetOutput();
}

void UserInterface::Mask::Invert()
{
    char* data = (char*)this->buffer->GetScalarPointer();
    std::transform(data, data + this->buffer->GetNumberOfPoints(), data, [](char value) {return (value == 0 ? 1 : 0); });
    this->buffer->Modified();
}

void UserInterface::slot_SaveRegions()
{
    _SaveBookmarks();
}

void UserInterface::slot_SaveROI()
{
    QFile file("ROI.txt");
    MacroOpenQFileToWrite(file);
    QDataStream stream(&file);

    if (_Is2DMode())
    {
        auto p1coord = m_Border->GetBorderRepresentation()->GetPositionCoordinate();
        auto p2coord = m_Border->GetBorderRepresentation()->GetPosition2Coordinate();
        double p1[3] = { 0,0,0 }, p2[3] = { 10,10,10 };
        p1coord->GetValue(p1);
        p2coord->GetValue(p2);
        stream << p1[0] << p1[1] << p1[2];
        stream << p2[0] << p2[1] << p2[2];
    }
    else
    {
        double maskBounds[6] = { 0,10,0,10,0,10 };
        vtkNew<vtkPlanes> planes;
        m_Box->GetPlanes(planes);
        vtkNew<vtkPolyData> boxPd;
        m_Box->GetPolyData(boxPd);
        boxPd->GetBounds(maskBounds);
        stream << maskBounds[0] << maskBounds[1] << maskBounds[2] << maskBounds[3] << maskBounds[4] << maskBounds[5];
    }
    file.close();
}
void UserInterface::slot_LoadROI()
{
    QFile file("ROI.txt");
    MacroOpenQFileToRead(file);
    QDataStream stream(&file);
    double p1[3] = { 0,0,0 }, p2[3] = { 10,10,10 };
    stream >> p1[0] >> p1[1] >> p1[2];
    stream >> p2[0] >> p2[1] >> p2[2];
    file.close();
    std::cout << "p1 = " << p1[0] << ", " << p1[1] << ", " << p1[2] << std::endl;
    std::cout << "p2 = " << p2[0] << ", " << p2[1] << ", " << p2[2] << std::endl;
    if (_Is2DMode())
    {
        m_Border->GetBorderRepresentation()->SetPosition(p1);
        m_Border->GetBorderRepresentation()->SetPosition2(p2);
        m_Border->Modified();
    }
    else
    {
        double maskBounds[6] = { p1[0], p1[1], p1[2], p2[0], p2[1], p2[2] };
        m_Box->SetPlaceFactor(1.0);
        m_Box->PlaceWidget(maskBounds);
        m_Box->Modified();
    }
    m_ui.RenWidget->update();
}

void UserInterface::_RemoveStructureFromVolume(ImagePtr structureMask, ImagePtr volume, bool invertMask)
{
    MacroAssert(structureMask->GetScalarType() == VTK_CHAR);
    //MacroAssert(volume->GetScalarType() == VTK_FLOAT);
    char* mask = (char*)structureMask->GetScalarPointer();
    int stride = volume->GetNumberOfScalarComponents();

    ushort* ushort_input = (ushort*)volume->GetScalarPointer();
    uchar* uchar_input = (uchar*)volume->GetScalarPointer();
    const int type = volume->GetScalarType();

    float* disp = nullptr;
    if(m_Input2D)
        disp = (float*)m_Input2D->GetScalarPointer();

    vtkIdType sz = volume->GetNumberOfPoints();
    for (vtkIdType i = 0; i < sz; ++i)
    {
        bool flag = (mask[i] != char(0));
        flag = invertMask ? !flag : flag;
        if (flag)
        {
            for (int k = 0; k < stride; ++k)
            {
                if (type == VTK_UNSIGNED_SHORT)
                    ushort_input[i*stride + k] = ushort(0);
                else
                    uchar_input[i*stride + k] = uchar(0);

                if (_Is2DMode() && disp)
                    disp[i*stride + k] = 0.0f;
            }
        }
    }
    volume->Modified();
    /*
    vtkNew<vtkImageCast> cast;
    cast->SetInputData(m_InputImage);
    cast->SetOutputScalarTypeToUnsignedChar();
    cast->Update();
    vtkNew<vtkPNGWriter> writer;
    writer->SetInputData(cast->GetOutput());
    writer->SetFileName("updated_input.png");
    writer->SetFileDimensionality(2);
    writer->Write();
    MacroPrint(writer->GetErrorCode());
    */
}

void UserInterface::slot_NodeChanged(size_t id)
{
#ifdef HIERARCHICAL
    if (id < m_structures.size())
    {
        m_InputImage->DeepCopy(m_structures[id].node);
        if(_Is2DMode())
            m_Input2D->DeepCopy(m_structures[id].node);
    }
#endif
}

void UserInterface::slot_showHistogramViewer(bool checked)
{
    m_ui.wHistogramViewer->setVisible(true);
}

void UserInterface::slot_ClippingChanged()
{
    // X-Axis
    QQuickWidget* x_axis = m_ui.wROI->findChild<QQuickWidget*>(QString("x_axis"));
    auto obj = x_axis->rootObject();
    float clip_values[6] = { 0,1,0,1,0,1 };
    clip_values[0] = QQmlProperty::read(obj, "first.value").toFloat();
    clip_values[3] = QQmlProperty::read(obj, "second.value").toFloat();
    //QQmlProperty::write(obj, "second.value", 0.5);

    QQuickWidget* y_axis = m_ui.wROI->findChild<QQuickWidget*>(QString("y_axis"));
    obj = y_axis->rootObject();
    clip_values[1] = QQmlProperty::read(obj, "first.value").toFloat();
    clip_values[4] = QQmlProperty::read(obj, "second.value").toFloat();

    QQuickWidget* z_axis = m_ui.wROI->findChild<QQuickWidget*>(QString("z_axis"));
    obj = z_axis->rootObject();
    clip_values[2] = QQmlProperty::read(obj, "first.value").toFloat();
    clip_values[5] = QQmlProperty::read(obj, "second.value").toFloat();

    m_ui.wVolumeVisualizer->ClipVolume(clip_values);
}

void UserInterface::UpdateVisualizer()
{
    m_ui.wVolumeVisualizer->SetImage(m_InputImage);
}

void UserInterface::slot_AddBookmark()
{
    int id = m_ui.wTreeView->GetSelectedItem();
    m_ui.wBookmarkTool->AddBookmark(m_structures[id].bookmark.GetRegion());
}
