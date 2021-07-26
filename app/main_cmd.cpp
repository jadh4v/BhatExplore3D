#include "vtkAutoInit.h" 
VTK_MODULE_INIT(vtkRenderingOpenGL2); // VTK was built with vtkRenderingOpenGL2
VTK_MODULE_INIT(vtkInteractionStyle);
//VTK_MODULE_INIT(vtkImagingCore);

//STD
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <set>
#include <vector>
//#include <conio.h>
#include <future>

//Qt
#include <QApplication>
#include <QImage>
#include <QLabel>
#include <QElapsedTimer>
#include <QDir>
#include <QTimer>
#include <QSurfaceFormat>
#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>

//VTK
#include <vtkNew.h>
//#include <vtkOutlineFilter.h>
//#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageCast.h>
#include <vtkImageExtractComponents.h>
#include <vtkImageResize.h>
#include <vtkContourFilter.h>
#include <vtkImageActor.h>
#include <vtkNamedColors.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkImageInterpolator.h>
#include <vtkImageMapper.h>
#include <vtkImageMapper3D.h>
#include <vtkActor2D.h>
#include <vtkProperty.h>
#include <vtkImageShiftScale.h>
#include <vtkCleanPolyData.h>
#include <vtkImageGradient.h>
#include <vtkImageNormalize.h>
#include <vtkImageDivergence.h>
#include <vtkImageLaplacian.h>
#include <vtkImageMathematics.h>
#include <vtkMetaImageReader.h>
#include <vtkMetaImageWriter.h>
#include <vtkTIFFReader.h>
#include <vtkTIFFWriter.h>
#include <vtkPointData.h>
#include <vtkImageMedian3D.h>
#include <vtkImagePlaneWidget.h>
#include <vtkBoxWidget.h>
#include <vtkImageCroppingRegionsWidget.h>
#include <vtkImageReslice.h>
#include <vtkROIStencilSource.h>
#include <vtkImageStencilToImage.h>
#include <vtkImageStencilData.h>
#include <vtkImageNoiseSource.h>
#include <vtkTransform.h>
#include <vtkLinearTransform.h>

//#include <QVTKRenderWidget.h>
#include <vtkImageViewer.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>


//Proj
//#include "utils/utils.h"
#include "ds/Bitvector.h"
#include "ds/Grid.h"
#include "ds/GridPoint.h"
#include "voldef/AlgoCudaDistanceField.h"
#include "bhat/Region.h"
#include "bhat/FullRegion.h"
#include "bhat/FeatureSpace.h"
#include "FeatureSpaceCuda.h"
#include "bhatt_global.h"
#include "io/VolumeReader.h"
#include "BhattParameters.h"
#include "ROIBoxCallback.h"
#include "RenderCallback.h"
#include "UserInterface.h"
#include "ActiveContourBlock.h"
#include "ProjUtils.h"
#include "utils/utils.h"


typedef float PrecType;
using std::cout;
using std::endl;
using std::vector;
using sjDS::Bitvector;

std::fstream gLog("./log.txt", std::fstream::out );
BhattParameters gParam;

vtkSmartPointer<vtkImageData> ReadImage(const QString& filename, bool smoothing=false)
{
    vtkSmartPointer<vtkImageData> ret = nullptr;
    
    VolumeReader reader(filename);
    if (reader.Read())
        ret = reader.GetVtkOutput();
    else
        return nullptr;

    // validate vtkImageReader2's output since it has no error code to provide
    if(ret)
    {
        int dim[3];
        ret->GetDimensions(dim);
        if (dim[0] <= 1)
            return nullptr;
    }

    // pre-Process the input image
    // RESIZE the image if down-sample factor is specified.
    int f = gParam.DownSampleFactor();
    if (f != 1)
    {
        int dim[3];
        ret->GetDimensions(dim);
        vtkNew<vtkImageResize> resize;
        resize->SetInputData(ret);
        resize->SetOutputDimensions(dim[0] / f, dim[1] / f, dim[2] / f);
        resize->Update();
        ret = resize->GetOutput();
    }

    const int numOfComps  = ret->GetNumberOfScalarComponents();
    const int numOfPoints = ret->GetNumberOfPoints();
    if (numOfComps == 1)
    {
        double range[2];
        ret->GetScalarRange(range);
        double range_width = range[1] - range[0];
        std::cout << "Original Scalar Range = " << range[0] << ", " << range[1] << std::endl;
        MacroPrint(range_width);

        vtkNew<vtkImageShiftScale> shift;
        shift->SetInputData(ret);
        shift->SetOutputScalarTypeToUnsignedShort();
        shift->SetShift(-range[0]);
        //double scale = gParam.ScalarRange() / (range[1] - range[0]);
        double scale = 1.0;
        if (range_width >= 0xffff)
            scale = double(0xffff) / range_width;

        shift->SetScale(scale);
        shift->ClampOverflowOn();
        shift->Update();
        ret = shift->GetOutput();

        if (smoothing)
        {
            vtkNew<vtkImageMedian3D> median;
            median->SetInputData(ret);
            median->SetKernelSize(3, 3, 3);
            median->Update();
            ret = median->GetOutput();
        }

        // move minimum value to 1.0 since 0 value has special meaning (for blanking voxels).
        ushort* scalars = static_cast<ushort*>(ret->GetScalarPointer());
        std::transform(scalars, (scalars + numOfComps * numOfPoints), scalars, [](ushort value) { constexpr ushort one = static_cast<ushort>(1); return (value < one ? one : value); });
        ret->Modified();
    }
    else if(numOfComps == 3)
    {
        // this maybe an rgb image, just typecast to float and proceed.
        vtkNew<vtkImageCast> cast;
        cast->SetInputData(ret);
        cast->SetOutputScalarTypeToUnsignedChar();
        cast->ClampOverflowOn();
        cast->Update();
        ret = cast->GetOutput();

        // move minimum value to 1.0 since 0 value has special meaning (for blanking voxels).
        uchar* scalars = static_cast<uchar*>(ret->GetScalarPointer());
        std::transform(scalars, (scalars + numOfComps * numOfPoints), scalars, [](uchar value) { constexpr uchar one = static_cast<uchar>(1); return (value < one ? one : value); });
        ret->Modified();
    }
    else
    {
        MacroFatalError("Unhandled number of components in input data: " << ret->GetNumberOfScalarComponents());
    }

    return ret;
}

vtkSmartPointer<vtkImageData> ReadInputMask(std::string filename)
{
    vtkNew<vtkMetaImageReader> reader;
    reader->SetFileName(filename.c_str());
    vtkNew<vtkImageExtractComponents> extractComp;
    extractComp->SetInputConnection(reader->GetOutputPort());
    extractComp->SetComponents(0);
    extractComp->Update();
    MacroAssert(reader->GetErrorCode() == 0);

    int dim[3];
    extractComp->GetOutput()->GetDimensions(dim);

    vtkNew<vtkImageInterpolator> inter;
    inter->SetBorderModeToClamp();
    inter->SetInterpolationModeToNearest();
    vtkNew<vtkImageResize> resize;
    resize->SetInputData(extractComp->GetOutput());
    resize->SetInterpolator(inter);
    resize->InterpolateOn();
    int f = gParam.DownSampleFactor();
    resize->SetOutputDimensions(dim[0] / f, dim[1] / f, dim[2] / f);
    resize->Update();

    double range[2];
    resize->GetOutput()->GetScalarRange(range);

    vtkNew<vtkImageShiftScale> shift;
    shift->SetInputData(resize->GetOutput());
    shift->SetOutputScalarTypeToUnsignedChar();
    shift->SetShift(range[0] * -1.0);
    shift->SetScale(gParam.ScalarRange()/(range[1] - range[0]));
    shift->Update();

    Utils::PrintScalarRange(shift->GetOutput(), "Input Mask");
    return shift->GetOutput();
}


void reComputeSpacing(vtkImageData* inputImage)
{
    double spacing[3];
    inputImage->GetSpacing(spacing);
    double maxValue = *std::max_element(spacing, spacing + 3);
    for (int i = 0; i < 3; ++i)
        spacing[i] /= maxValue;

    //spacing[0] = 1.0;
    //spacing[1] = 1.0;
    //spacing[2] = 1.0;
    inputImage->SetSpacing(spacing);
}

void UpdateLogDestination(const QString& inputFileName)
{
    QString logfile = inputFileName;
    int pos = std::max(inputFileName.lastIndexOf('/'), inputFileName.lastIndexOf('\\'));
    logfile.truncate(pos);
    logfile += "/log.txt";
    gLog = std::fstream(logfile.toLatin1().constData(), std::fstream::out);
}

int main(int argc, char** argv)
{
    cudaSetDevice(0);
    // set surface format before application initialization
    //auto surface = QVTKRenderWidget::defaultFormat();
    //surface.setSamples(8);
    //QSurfaceFormat::setDefaultFormat(surface);
    QApplication app(argc, argv);
    app.setStyleSheet("QTreeView::branch:has-siblings:!adjoins-item {"
    "border-image: url(vline.png) 0;"
    "}"
        "QTreeView::branch:has-siblings:adjoins-item{"
            "border-image: url(branch-more.png) 0;"
    "}"
        "QTreeView::branch:!has-children:!has-siblings:adjoins-item {"
            "border-image: url(branch-end.png) 0;"
    "}"
        "QTreeView::branch:has-children:!has-siblings:closed,"
        "QTreeView::branch:closed:has-children:has-siblings {"
                "border-image: none;"
                "image: url(branch-closed.png);"
    "}"
        "QTreeView::branch:open:has-children:!has-siblings,"
        "QTreeView::branch:open:has-children:has-siblings {"
                "border-image: none;"
                "image: url(branch-open.png);"
    "}");

    MacroConfirmOrReturn(argc >= 9, 0);
    int numOfBlocks = QString(argv[1]).toInt();
    double alpha = QString(argv[2]).toDouble();
    double stepSize = QString(argv[3]).toDouble();
    double del = QString(argv[4]).toDouble();
    double narrow_band = QString(argv[5]).toDouble();
    int num_of_iterations = QString(argv[6]).toInt();
    int recomputePhiIterations = QString(argv[7]).toInt();
    int downSampleFactor = QString(argv[8]).toInt();
    QString filename;
    if(argc > 9)
    filename = QString(argv[9]);

    MacroPrint(numOfBlocks);
    MacroPrint(alpha);
    MacroPrint(stepSize);
    MacroPrint(del);
    MacroPrint(narrow_band);
    MacroPrint(num_of_iterations);
    MacroPrint(recomputePhiIterations);
    MacroPrint(downSampleFactor);

    //gParam.SetParameters(0.1, 0.1, 2.0, 2.0, 100000, 100, 1); //  works for vp11 kidneys, lesion, areteries.
    //gParam.SetParameters(0.1, 0.5, 2.0, 2.0, 100000, 10, 1); //  works for vp11 kidneys, lesion, areteries.
    //gParam.SetParameters(0.4, 3.0, 2.0, 2.0, 100000, 10, 1); //  works for vp11 kidneys, lesion, areteries.
    gParam.SetParameters(alpha, stepSize, del, narrow_band, num_of_iterations, recomputePhiIterations, downSampleFactor);

// read input image and mask objects
    vtkSmartPointer<vtkImageData> inputImage = nullptr;
    vtkSmartPointer<vtkImageData> displayImage  = nullptr;
    vtkSmartPointer<vtkImageData> inputOrig  = nullptr;

    QMessageBox::StandardButton reply = QMessageBox::question(nullptr, "Load dataset", "Load command-line dataset?", QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::No)
        filename.clear();
    if (filename.isEmpty())
    {
        filename = QFileDialog::getOpenFileName(0, "Open File", "F:/ImagingData/bhat/tooth/");
        if (filename.isEmpty())
            filename = QFileDialog::getExistingDirectory(0, "Open DIR", "F:/ImagingData/");
    }

    UpdateLogDestination(filename);

    bool smooth = true;
    inputImage = ReadImage(filename, smooth);
    if (!inputImage.Get())
        MacroFatalError("Cannot read input datafile.");

    reComputeSpacing(inputImage);
    gParam.SetGlobalRange(inputImage->GetScalarRange());
    UserInterface ui(inputImage, displayImage);
    ui.SetNumberOfBlocks(numOfBlocks);
    ui.show();
    ui.UpdateVisualizer();
    return app.exec();
}
