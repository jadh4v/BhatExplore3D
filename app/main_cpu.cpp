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
#include <conio.h>
#include <future>

//Qt
#include <QApplication>
#include <QImage>
#include <QLabel>
#include <QElapsedTimer>
#include <QDir>
#include <QTimer>

//VTK
#include <vtkNew.h>
#include <vtkOutlineFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageCast.h>
#include <vtkBMPReader.h>
#include <vtkCamera.h>
#include <vtkImageExtractComponents.h>
#include <vtkImageResize.h>
#include <vtkImplicitModeller.h>
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
#include <vtkPointData.h>
#include <vtkImageMedian3D.h>
#include <vtkImagePlaneWidget.h>
#include <vtkBoxWidget.h>
#include <vtkImageCroppingRegionsWidget.h>
#include <vtkImageReslice.h>
#include <vtkPNGReader.h>
#include <vtkROIStencilSource.h>
#include <vtkImageStencilToImage.h>
#include <vtkImageStencilData.h>
#include <vtkImageNoiseSource.h>
#include <vtkTransform.h>
#include <vtkLinearTransform.h>

//Proj
//#include "utils/utils.h"
#include "ds/Bitvector.h"
#include "ds/Grid.h"
#include "ds/GridPoint.h"
#include "voldef/AlgoCudaDistanceField.h"
#include "MainWindow.h"
#include "bhat/Region.h"
#include "bhat/FullRegion.h"
#include "bhat/FeatureSpace.h"
#include "bhat/FeatureSpaceCuda.h"
#include "bhatt_global.h"
#include "io/VolumeReader.h"
#include "BhattParameters.h"
#include "ROIBoxCallback.h"
#include "RenderCallback.h"

typedef unsigned int uint;
typedef float PrecType;
typedef Bhat::FeatureSpace<1, PrecType> Space;
//typedef Bhat::FeatureSpaceCuda<1, PrecType> SpaceCuda;
using std::cout;
using std::endl;
using std::vector;
using sjDS::Bitvector;

std::fstream gLog("./log.txt", std::fstream::out );
BhattParameters gParam;


void PrintScalarRange(vtkImageData* volume, const char* volume_name)
{
    double r[2] = { 0,0 };
    volume->GetScalarRange(r);
    std::cout << volume_name << " scalar_range: [" << r[0] << ", " << r[1] << "]" << std::endl;
}


vtkSmartPointer<vtkImageData> construct_phi(vtkImageData* mask)
{
    vtkNew<vtkContourFilter> contourFilter;
    contourFilter->SetInputData(mask);
    contourFilter->SetNumberOfContours(1);
    contourFilter->SetValue(0, 127);
    contourFilter->Update();
    auto contour = contourFilter->GetOutput();
    std::cout << "Contour size = " << contour->GetNumberOfPoints() << std::endl;
    vtkNew<vtkCleanPolyData> clean;
    clean->SetInputData(contour);
    clean->Update();
    contour = clean->GetOutput();
    std::cout << "Clean Contour size = " << contour->GetNumberOfPoints() << std::endl;

    vtkNew<vtkImageData> dist;
    dist->SetDimensions(mask->GetDimensions());
    dist->SetExtent(mask->GetExtent());
    dist->AllocateScalars(VTK_FLOAT, 1);

    float* dist_ptr =  (float*)dist->GetScalarPointer();
    for (int i = 0; i < dist->GetNumberOfPoints(); ++i)
    {
        double x[3];
        dist->GetPoint(i, x);
        double minDist = 1e10;
        for (int j = 0; j < contour->GetNumberOfPoints(); ++j)
        {
            double c[3];
            contour->GetPoint(j, c);
            vtkMath::Subtract(x, c, c);
            double d = vtkMath::Norm(c);
            minDist = std::min(d, minDist);
        }
        if (minDist < 0.5) minDist = 0.0;
        dist_ptr[i] = minDist;
    }

    MacroAssert(mask->GetScalarType() == VTK_UNSIGNED_CHAR);
    MacroAssert(mask->GetNumberOfPoints() == dist->GetNumberOfPoints());
    uchar* mask_ptr = (uchar*)mask->GetScalarPointer();
    for (int i = 0; i < dist->GetNumberOfPoints(); ++i)
    {
        if(mask_ptr[i] != 0)
            dist_ptr[i] *= -1.0;
    }

    double range[2];
    dist->GetScalarRange(range);
    //Utils::PrintScalarRange(dist, "dist");
 
    vtkNew<vtkNamedColors> colors;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> window;
    vtkNew<vtkRenderWindowInteractor> interactor;
    vtkNew<vtkActor> contourActor;
    vtkNew<vtkPolyDataMapper> contourMapper;
    contourMapper->SetInputData(contour);
    contourActor->SetMapper(contourMapper);
    contourActor->GetProperty()->SetColor(colors->GetColor3d("LightRed").GetData());
    renderer->AddActor(contourActor);

    if (mask->GetDimensions()[2] > 1)
    {
        vtkNew<vtkOutlineFilter> outliner;
        outliner->SetInputData(mask);
        outliner->Update();
        auto outline = outliner->GetOutput();
        {
            vtkNew<vtkActor> actor;
            vtkNew<vtkPolyDataMapper> mapper;
            mapper->SetInputData(outline);
            actor->SetMapper(mapper);
            renderer->AddActor(actor);
        }
        renderer->ResetCamera();
        renderer->SetBackground(colors->GetColor3d("Burlywood").GetData());
        window->AddRenderer(renderer);
        interactor->SetRenderWindow(window);
        vtkNew<vtkInteractorStyleTrackballCamera> style;
        interactor->SetInteractorStyle(style);
        interactor->Start();
    }
    else
    {
        vtkNew<vtkImageShiftScale> shift;
        shift->SetInputData(dist);
        shift->SetOutputScalarTypeToUnsignedChar();
        shift->SetShift(range[0] * -1.0);
        shift->SetScale((range[1] - range[0])*255.0);
        shift->Update();
        auto draw = shift->GetOutput();
        draw->GetScalarRange(range);

        // Create an actor
        vtkNew<vtkImageActor> imageActor;
        imageActor->GetMapper()->SetInputData(dist);
        // Setup renderer

        contourActor->GetProperty()->SetLineWidth(2.0);

        renderer->AddActor(imageActor);
        renderer->ResetCamera();
        renderer->SetBackground(colors->GetColor3d("Burlywood").GetData());
        window->AddRenderer(renderer);
        interactor->SetRenderWindow(window);
        vtkNew<vtkInteractorStyleTrackballCamera> style;
        interactor->SetInteractorStyle(style);
        interactor->Start();
    }
    return dist;
}

void construct_phi_cuda(const Bitvector& mask, const Bhat::AbstractRegion& baseRegion, float* oPhi)
{
    // Compute distance field given the mask volume.
    voldef::AlgoCudaDistanceField ds;
    auto domain = baseRegion.Points();
    ds.SetDomainPoints(domain.data(), domain.size());
    auto object = baseRegion.SubRegionBoundaryPoints(mask);
    ds.SetObjectPoints(object.data(), object.size());

    QElapsedTimer timer;
    timer.start();
    ds.Run();
    std::cout << "ds.Run() time = " << timer.elapsed() << " \n";

    vector<float> distances = ds.GetOutput();
    auto r = std::minmax_element(distances.begin(), distances.end());
    cout << "distances { " << *(r.first) << ", " << *(r.second) << "} " << endl;

    // Set negative sign for internal points.
    for (size_t i = 0; i < distances.size(); ++i)
    {
        if (mask.Get(i))
            distances[i] *= -1.0;
    }
    r = std::minmax_element(distances.begin(), distances.end());
    cout << "distances { " << *(r.first) << ", " << *(r.second) << "} " << endl;

    // copy output to a vtkImageData
    std::memcpy(oPhi, distances.data(), sizeof(float)*distances.size());

    //Debug Display
#if 0
    vtkNew<vtkContourFilter> contourFilter;
    contourFilter->SetInputData(mask);
    contourFilter->SetNumberOfContours(1);
    contourFilter->SetValue(0, 0.5);
    contourFilter->Update();
    auto contour = contourFilter->GetOutput();
    //MacroPrint(contour->GetNumberOfPoints());
    mask->GetScalarRange(range);
    std::cout << "mask range = " << range[0] << ", " << range[1] << std::endl;
    oPhi->GetScalarRange(range);
    std::cout << "381: oPhi range = " << range[0] << ", " << range[1] << std::endl;

    vtkNew<vtkNamedColors> colors;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> window;
    vtkNew<vtkRenderWindowInteractor> interactor;
    vtkNew<vtkActor> contourActor;
    vtkNew<vtkPolyDataMapper> contourMapper;
    contourMapper->SetInputData(contour);
    contourActor->SetMapper(contourMapper);
    contourActor->GetProperty()->SetColor(colors->GetColor3d("LightRed").GetData());
    renderer->AddActor(contourActor);

    if (mask->GetDimensions()[2] > 1)
    {
        vtkNew<vtkOutlineFilter> outliner;
        outliner->SetInputData(mask);
        outliner->Update();
        auto outline = outliner->GetOutput();
        {
            vtkNew<vtkActor> actor;
            vtkNew<vtkPolyDataMapper> mapper;
            mapper->SetInputData(outline);
            actor->SetMapper(mapper);
            renderer->AddActor(actor);
        }
        renderer->ResetCamera();
        renderer->SetBackground(colors->GetColor3d("Burlywood").GetData());
        window->AddRenderer(renderer);
        interactor->SetRenderWindow(window);
        vtkNew<vtkInteractorStyleTrackballCamera> style;
        interactor->SetInteractorStyle(style);
        interactor->Start();
    }
/*
    else
    {
        vtkNew<vtkImageShiftScale> shift;
        shift->SetInputData(dist);
        shift->SetOutputScalarTypeToUnsignedChar();
        shift->SetShift(range[0] * -1.0);
        shift->SetScale((range[1] - range[0])*255.0);
        shift->Update();
        auto draw = shift->GetOutput();
        draw->GetScalarRange(range);

        // Create an actor
        vtkNew<vtkImageActor> imageActor;
        imageActor->GetMapper()->SetInputData(dist);
        // Setup renderer

        contourActor->GetProperty()->SetLineWidth(2.0);

        renderer->AddActor(imageActor);
        renderer->ResetCamera();
        renderer->SetBackground(colors->GetColor3d("Burlywood").GetData());
        window->AddRenderer(renderer);
        interactor->SetRenderWindow(window);
        vtkNew<vtkInteractorStyleImage> style;
        interactor->SetInteractorStyle(style);
        interactor->Start();
    } */
#endif
}

void convert_to_vtkmask(const sjDS::Bitvector& mask, vtkSmartPointer<vtkImageData> vtk_mask)
{
    float* vtk_mask_ptr = (float*)vtk_mask->GetScalarPointer();
    for (vtkIdType i = 0; i < vtk_mask->GetNumberOfPoints(); ++i)
    {
        if (mask.Get(i))
            vtk_mask_ptr[i] = 1.0f;
        else
            vtk_mask_ptr[i] = 0.0f;
    }
}

void construct_mask_vector(vtkImageData* phi, const Bhat::Region& baseRegion, sjDS::Bitvector& mask)
{
    mask.ClearBits();
    size_t numOfVoxels = (size_t)phi->GetNumberOfPoints();
    auto phi_data  = phi->GetPointData()->GetScalars();

    for (int i = 0; i < numOfVoxels; ++i)
    {
        if (baseRegion.Contains(i))
        {
            if (phi_data->GetVariantValue(i).ToDouble() < 0)
                mask.Set(i);
        }
    }
}

vtkSmartPointer<vtkRenderWindow> show_progress(vtkSmartPointer<vtkImageData> inputImage, vtkSmartPointer<vtkImageData> phi)
{
    vtkNew<vtkContourFilter> contourFilter;
    contourFilter->SetInputData(phi);
    contourFilter->SetNumberOfContours(1);
    contourFilter->SetValue(0, 0.0);

    // Create an actor
    vtkNew<vtkImageActor> imageActor;
    imageActor->GetMapper()->SetInputData(inputImage);

    // Setup renderer
    vtkNew<vtkNamedColors> colors;

    vtkNew<vtkActor> contourActor;
    {
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(contourFilter->GetOutputPort());
        contourActor->SetMapper(mapper);
        contourActor->GetProperty()->SetColor(colors->GetColor3d("Red").GetData());
        contourActor->GetProperty()->SetLineWidth(2.0);
    }

    vtkNew<vtkRenderer> renderer;
    renderer->AddActor(imageActor);
    renderer->AddActor(contourActor);
    renderer->ResetCamera();
    renderer->SetBackground(colors->GetColor3d("Burlywood").GetData());

    // Setup render window
    vtkNew<vtkRenderWindow> window;
    window->AddRenderer(renderer);
    window->SetPosition(500, 500);

    // Setup render window interactor
    vtkNew<vtkRenderWindowInteractor> interactor;
    interactor->SetRenderWindow(window);
    // Setup interactor style (this is what implements the zooming, panning and brightness adjustment functionality)
    vtkNew<vtkInteractorStyleImage> style;
    interactor->SetInteractorStyle(style);
    window->Render();

    // Initialize must be called prior to creating timer events.
    interactor->Initialize();

    // Sign up to receive TimerEvent
    vtkNew<RenderCallback> cb;
    cb->SetWindow(window);
    //cb->SetActor(contourActor);
    interactor->AddObserver(vtkCommand::TimerEvent, cb);
    int timerId = interactor->CreateRepeatingTimer(2000);

    interactor->Start();

    return window;
}

vtkSmartPointer<vtkRenderWindow> show_progress_3D(vtkSmartPointer<vtkImageData> inputImage, vtkSmartPointer<vtkImageData> phi)
{
    vtkNew<vtkContourFilter> contourFilter;
    contourFilter->SetInputData(phi);
    contourFilter->SetNumberOfContours(1);
    contourFilter->SetValue(0, 0.0);

    // Setup renderer
    vtkNew<vtkNamedColors> colors;
    vtkNew<vtkRenderer> renderer;
    {
        vtkNew<vtkActor> contourActor;
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(contourFilter->GetOutputPort());
        contourActor->SetMapper(mapper);
        contourActor->GetProperty()->SetColor(colors->GetColor3d("Red").GetData());
        contourActor->GetProperty()->SetOpacity(0.3);
        contourActor->GetProperty()->SetLineWidth(2.0);
        renderer->AddActor(contourActor);
        //contourActor->VisibilityOff();
    }
    vtkNew<vtkOutlineFilter> outliner;
    outliner->SetInputData(phi);
    outliner->Update();
    auto outline = outliner->GetOutput();
    vtkNew<vtkActor> outlineActor;
    {
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputData(outline);
        outlineActor->SetMapper(mapper);
        renderer->AddActor(outlineActor);
        //outlineActor->VisibilityOff();
    }

    //renderer->ResetCamera();
    renderer->SetBackground(colors->GetColor3d("Black").GetData());

    // Setup render window
    vtkNew<vtkRenderWindow> window;
    window->AddRenderer(renderer);
    window->SetPosition(900, 0);
    window->SetMultiSamples(4);
    window->SetSize(1000, 1000);

    // Setup render window interactor
    vtkNew<vtkRenderWindowInteractor> interactor;
    interactor->SetRenderWindow(window);
    // Setup interactor style (this is what implements the zooming, panning and brightness adjustment functionality)
    vtkNew<vtkInteractorStyleTrackballCamera> style;
    interactor->SetInteractorStyle(style);

    /*
    //Add Image cropping widget (roi widget)
    vtkNew<vtkBoxWidget> box;
    box->SetInteractor(interactor);
    box->SetProp3D(outlineActor);
    box->PlaceWidget();
    vtkNew<ROIBoxCallback> boxCallback;
    box->AddObserver(vtkCommand::InteractionEvent, boxCallback);
    box->On();

    // Add imageplanewidget
    vtkNew<vtkImagePlaneWidget> planeWidget;
    planeWidget->SetInputData(inputImage);
    planeWidget->TextureVisibilityOn();
    planeWidget->TextureInterpolateOn();
    planeWidget->UpdatePlacement();
    planeWidget->SetPlaneOrientation(1);
    //planeWidget->SetWindowLevel(168, 162);
    //planeWidget->SetWindowLevel(347, 16);
    planeWidget->SetWindowLevel(250, 85);
    //planeWidget->SetInteractor(window->GetInteractor());
    planeWidget->PlaceWidget();
    */

    window->Render();
    interactor->Initialize();
    window->Render();
    /*
    planeWidget->On();
    planeWidget->SetSliceIndex(60);
    //renderer->GetActiveCamera()->SetPosition(0, -1, 1);
    renderer->GetActiveCamera()->SetPosition(0, -1, 0);
    renderer->GetActiveCamera()->SetFocalPoint(inputImage->GetOrigin());
    renderer->GetActiveCamera()->SetViewUp(0, 0, 1);
    renderer->ResetCamera();

    // Initialize must be called prior to creating timer events.
    //interactor->Initialize();
    */

    // Sign up to receive TimerEvent
    vtkNew<RenderCallback> cb;
    cb->SetWindow(window);
    //cb->SetActor(contourActor);
    interactor->AddObserver(vtkCommand::TimerEvent, cb);
    int timerId = interactor->CreateRepeatingTimer(2000);

    // Render and start interaction
    interactor->Start();

    return window;
}

inline double img_access(vtkImageData* img, int i, int j, int k)
{
    return img->GetScalarComponentAsDouble(i, j, k, 0);
}

void compute_K(vtkSmartPointer<vtkImageData> K, vtkSmartPointer<vtkImageData> phi, double alpha)
{
    int dim[3] = { 0,0,0 };
    phi->GetDimensions(dim);
    int dim_k[3] = { 0,0,0 };
    K->GetDimensions(dim_k);
    MacroAssert(dim[0] == dim_k[0]);
    MacroAssert(dim[1] == dim_k[1]);
    MacroAssert(dim[2] == dim_k[2]);

    for (int z = 0; z < dim[2]; ++z)
    {
        int k = z;
        if(dim[2] > 1)
            k = vtkMath::ClampValue(z, 1, dim[2] - 2);

        for (int y = 0; y < dim[1]; ++y)
        {
            int j = vtkMath::ClampValue(y, 1, dim[1] - 2);
            for (int x = 0; x < dim[0]; ++x)
            {
                int i = vtkMath::ClampValue(x, 1, dim[0] - 2);
                double phi_x  = img_access(phi, i  , j+1, k  ) - img_access(phi, i  , j-1, k  );
                double phi_y  = img_access(phi, i+1, j  , k  ) - img_access(phi, i-1, j  , k  );

                double phi_xx = img_access(phi,i  ,j+1,k) - 2 * img_access(phi,i,j,k) + img_access(phi,i  ,j-1,k);
                double phi_yy = img_access(phi,i+1,j  ,k) - 2 * img_access(phi,i,j,k) + img_access(phi,i-1,j  ,k);

                double phi_xy = -0.25*img_access(phi,i-1,j-1,k) - 0.25*img_access(phi,i+1,j+1,k)
                    + 0.25*img_access(phi,i-1,j+1,k) + 0.25*img_access(phi,i+1,j-1,k);

                double norm = std::sqrt(phi_x*phi_x + phi_y*phi_y);
                double k_value = ((phi_x*phi_x*phi_yy + phi_y*phi_y*phi_xx - 2 * phi_x*phi_y*phi_xy) /
                    (pow((phi_x*phi_x + phi_y*phi_y + gParam.Eps()), 1.5)))*norm;

                K->SetScalarComponentFromDouble(x, y, z, 0, k_value*alpha);
            }
        }
    }
}

void PrintScalarType(vtkImageData* img, const char* img_name)
{
    cout << img_name << " scalar type: " << img->GetScalarTypeAsString() << endl;
}

void compute_K_div(vtkSmartPointer<vtkImageData> alphaK, vtkSmartPointer<vtkImageData> phi, double alpha)
{
    // Compute gradient of phi
    vtkNew<vtkImageGradient> gradientFilter;
    gradientFilter->SetInputData(phi);
    //gradientFilter->SetDimensionality(2);
    gradientFilter->SetNumberOfThreads(gParam.NumOfThreads());
    gradientFilter->Update();
    // normalize the gradients
    vtkNew<vtkImageNormalize> normalize;
    normalize->SetInputData(gradientFilter->GetOutput());
    normalize->SetNumberOfThreads(gParam.NumOfThreads());
    normalize->Update();
    auto N = normalize->GetOutput();

/*
    std::cout << "CPU gradient = " << std::endl;
    for (int i = 0; i < N->GetNumberOfPoints(); ++i)
    {
        double g[3];
        N->GetPointData()->GetScalars()->GetTuple(i, g);
        std::cout << "(" << g[0] << ", " << g[1] << ", " << g[2] << ") ";
    }
*/

    // compute divergence of normalized the gradients
    vtkNew<vtkImageDivergence> divFilter;
    divFilter->SetInputData(normalize->GetOutput());
    divFilter->SetNumberOfThreads(gParam.NumOfThreads());
    divFilter->Update();

    vtkNew<vtkImageMathematics> imageMath;
    imageMath->SetInputData(divFilter->GetOutput());
    imageMath->SetConstantK(alpha);
    imageMath->SetOperationToMultiplyByK();
    imageMath->SetNumberOfThreads(gParam.NumOfThreads());
    imageMath->Update();

    vtkNew<vtkImageCast> cast;
    cast->SetInputData(imageMath->GetOutput());
    cast->SetOutputScalarTypeToFloat();
    cast->SetNumberOfThreads(gParam.NumOfThreads());
    cast->Update();

/*
    auto Div = cast->GetOutput();
    std::cout << "CPU divergence = " << std::endl;
    for (int i = 0; i < Div->GetNumberOfPoints(); ++i)
    {
        float value = Div->GetPointData()->GetScalars()->GetVariantValue(i).ToFloat();
        std::cout << " " << value;
    }
*/
    alphaK->DeepCopy(cast->GetOutput());
}

vtkSmartPointer<vtkImageData> Read2DImage(std::string filename)
{
    vtkNew<vtkBMPReader> bmpReader;
    bmpReader->SetFileName(filename.c_str());
    vtkNew<vtkImageExtractComponents> extractComp;
    extractComp->SetInputConnection(bmpReader->GetOutputPort());
    extractComp->SetComponents(0);
    extractComp->Update();
    MacroAssert(bmpReader->GetErrorCode() == 0);
    return extractComp->GetOutput();
}

vtkSmartPointer<vtkImageData> ReadImageSeries(std::string path)
{
    QDir dir(path.c_str());
    if (!dir.exists())
    {
        MacroWarning("Cannot read path.");
        return nullptr;
    }

    auto files = dir.entryList(QDir::Files);
    vtkNew<vtkStringArray> fileNames;
    fileNames->SetNumberOfValues(files.size());
    QString qpath(path.c_str());
    for (int i=0; i < files.size(); ++i)
    {
        QString value = qpath + "/" + files[i];
        fileNames->SetValue(i, value.toLatin1().constData());
    }

    vtkNew<vtkPNGReader> reader;
    reader->SetFileNames(fileNames);
    reader->Update();
    return reader->GetOutput();
}

vtkSmartPointer<vtkImageData> ReadImage(std::string filename, bool mask=false, bool smoothing=false)
{
    vtkSmartPointer<vtkImageData> ret = nullptr;

    // Attempt 1, try our own reader that supports other formats like pvm.
    if (!ret)
    {
        VolumeReader reader(filename);
        if(reader.Read())
            ret = reader.GetVtkOutput();
    }

    // Attempt 2, maybe it is an image series.
    if(!ret)
        ret = ReadImageSeries(filename);

    // Attempt 1, try vtk's generalized image-reader
    if (!ret)
    {
        vtkNew<vtkImageReader2> reader2;
        reader2->SetFileName(filename.c_str());
        reader2->Update();
        ret = reader2->GetOutput();
    }

    // validate vtkImageReader2's output since it has no error code to provide
    if(ret)
    {
        int dim[3];
        ret->GetDimensions(dim);
        if (dim[0] <= 1)
            ret = nullptr;
    }

    // pre-Process the input image
    if (ret)
    {
        vtkNew<vtkImageExtractComponents> extractComp;
        extractComp->SetInputData(ret);
        extractComp->SetComponents(0);
        extractComp->Update();
        MacroAssert(extractComp->GetErrorCode() == 0);

        int dim[3];
        extractComp->GetOutput()->GetDimensions(dim);
        vtkNew<vtkImageResize> resize;
        resize->SetInputData(extractComp->GetOutput());
        int f = gParam.DownSampleFactor();
        resize->SetOutputDimensions(dim[0] / f, dim[1] / f, dim[2] / f);
        resize->Update();

        double range[2];
        resize->GetOutput()->GetScalarRange(range);

        vtkNew<vtkImageShiftScale> shift;
        shift->SetInputData(resize->GetOutput());

        if(mask)
            shift->SetOutputScalarTypeToUnsignedChar();

        shift->SetShift(range[0] * -1.0);
        shift->SetScale(255.0/(range[1] - range[0]));
        shift->Update();
        ret = shift->GetOutput();

        if (smoothing)
        {
            vtkNew<vtkImageMedian3D> median;
            median->SetInputData(shift->GetOutput());
            median->SetKernelSize(3, 3, 3);
            median->Update();
            ret = median->GetOutput();
        }
    }
    return ret;
}

vtkSmartPointer<vtkImageData> AddNoise(vtkSmartPointer<vtkImageData> input, const double noisePercentage)
{
    double input_range[2];
    input->GetScalarRange(input_range);
    const double inputScalarWindow = input_range[1] - input_range[0];
    const double noise_range[] = { -0.005*noisePercentage*inputScalarWindow, 0.005*noisePercentage*inputScalarWindow };
    vtkNew<vtkImageNoiseSource> noiseSource;
    noiseSource->SetMinimum(noise_range[0]);
    noiseSource->SetMaximum(noise_range[1]);
    noiseSource->SetWholeExtent(input->GetExtent());
    noiseSource->Update();
    MacroPrint(noiseSource->GetOutput()->GetScalarTypeAsString());

    vtkNew<vtkImageCast> cast1;
    cast1->SetInputData(noiseSource->GetOutput());
    cast1->SetOutputScalarTypeToDouble();
    cast1->Update();

    vtkNew<vtkImageCast> cast2;
    cast2->SetInputData(input);
    cast2->SetOutputScalarTypeToDouble();
    cast2->Update();

    vtkNew<vtkImageMathematics> imageMath;
    imageMath->SetInput1Data(cast1->GetOutput());
    imageMath->SetInput2Data(cast2->GetOutput());
    imageMath->SetOperationToAdd();
    imageMath->Update();

    double range[2];
    imageMath->GetOutput()->GetScalarRange(range);
    vtkNew<vtkImageShiftScale> shift;
    shift->SetInputData(imageMath->GetOutput());
    shift->SetShift(range[0] * -1.0);
    shift->SetScale(255.0/(range[1] - range[0]));
    shift->SetOutputScalarTypeToFloat();
    shift->Update();

    return shift->GetOutput();
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
    shift->SetScale(255.0/(range[1] - range[0]));
    shift->Update();

    PrintScalarRange(shift->GetOutput(), "Input Mask");
    return shift->GetOutput();
}

vtkSmartPointer<vtkImageData> ConstructFloatImage(int dim[3], double spacing[3])
{
    vtkNew<vtkImageData> img;
    img->SetDimensions(dim);
    img->SetOrigin(0, 0, 0);
    img->SetSpacing(spacing);
    img->AllocateScalars(VTK_FLOAT, 1);
    return img;
}

void WriteOutputMask(vtkImageData* vtk_mask, int dim[3])
{
    vtkNew<vtkImageInterpolator> inter;
    inter->SetInterpolationModeToNearest();

    voldef::AlgoCudaDistanceField ds(vtk_mask, true);
    ds.Run();
    vector<float> distances = ds.GetOutput();

    auto dist = ConstructFloatImage(vtk_mask->GetDimensions(), vtk_mask->GetSpacing());
    float* dist_ptr = (float*)dist->GetScalarPointer();
    std::copy(distances.begin(), distances.end(), dist_ptr);
    //std::memcpy(dist_ptr, distances.data(), sizeof(float)*distances.size());
    dist->Modified();
    dist->GetPointData()->Modified();
    dist->GetPointData()->GetScalars()->Modified();
    PrintScalarRange(dist, "dist_out");

    for (vtkIdType i = 0; i < dist->GetNumberOfPoints(); ++i)
    {
        if (dist_ptr[i] < -2.0f)
            dist_ptr[i] = 1.0;
        else
            dist_ptr[i] = 0.0;
    }
    dist->Modified();
    dist->GetPointData()->Modified();
    dist->GetPointData()->GetScalars()->Modified();

    PrintScalarRange(dist, "dist_out_mod");

    {
        vtkNew<vtkImageResize> resize;
        resize->SetInputData(vtk_mask);
        resize->SetOutputDimensions(dim);
        resize->InterpolateOn();
        resize->SetInterpolator(inter);
        resize->Update();
        vtkNew<vtkMetaImageWriter> writer;
        writer->SetInputData(resize->GetOutput());
        writer->SetFileName("seg_out.mhd");
        writer->Write();
    }

    {
        vtkNew<vtkImageResize> resize;
        resize->SetInputData(dist);
        resize->SetOutputDimensions(dim);
        resize->InterpolateOn();
        resize->SetInterpolator(inter);
        resize->Update();
        vtkNew<vtkMetaImageWriter> writer;
        writer->SetInputData(resize->GetOutput());
        writer->SetFileName("new_seed.mhd");
        writer->Write();
    }
}

vector<Space::Point> GetVoxels(vtkImageData* inputImage, const Bhat::AbstractRegion& baseRegion)
{
    size_t baseSize = baseRegion.Size();
    vector<Space::Point> points;
    points.reserve(baseSize);

    auto scalars = inputImage->GetPointData()->GetScalars();

    for (size_t i = 0; i < baseSize; ++i)
    {
        double voxelValue = scalars->GetVariantValue(baseRegion[i]).ToDouble();
        Space::Point x(voxelValue);
        points.push_back(x);
    }
    return points;
}
vector<PrecType> GetCudaVoxels(vtkImageData* inputImage, const Bhat::AbstractRegion& baseRegion)
{
    size_t baseSize = baseRegion.Size();
    vector<PrecType> points;
    points.reserve(baseSize);

    auto scalars = inputImage->GetPointData()->GetScalars();

    for (size_t i = 0; i < baseSize; ++i)
    {
        double voxelValue = scalars->GetVariantValue(baseRegion[i]).ToDouble();
        points.push_back(PrecType(voxelValue));
    }
    return points;
}

void ConstructMask(vtkImageData* inputMask, const Bhat::AbstractRegion& baseRegion, Bitvector& mask)
{
    mask.ClearBits();
    auto data = inputMask->GetPointData()->GetScalars();
    size_t baseSize = baseRegion.Size();
    for (size_t i = 0; i < baseSize; ++i)
    {
        if (data->GetVariantValue(baseRegion[i]).ToDouble() > 0)
        {
            mask.Set(i);
        }
    }
}

void ConstructMask(const float* phi, size_t sz, Bitvector& mask)
{
    mask.ClearBits();
    for (size_t i = 0; i < sz; ++i)
    {
        if (phi[i] <= 0)
            mask.Set(i);
    }
}

void UpdatePhiImage(const vector<float>& phi, const Bhat::AbstractRegion& baseRegion, vtkImageData* phi_image)
{
    if (phi.empty())
        return;

    float maxValue = * std::max_element(phi.begin(), phi.end());

    float* phi_image_data = (float*)phi_image->GetScalarPointer();
    for (vtkIdType i = 0; i < phi_image->GetNumberOfPoints(); ++i)
        phi_image_data[i] = maxValue;

    for (size_t i = 0; i < baseRegion.Size(); ++i)
        phi_image_data[baseRegion[i]] = phi[i];

    phi_image->Modified();
}

// 2D
//#define ZEBRA 1
//#define CHEETAH 1
//#define TRAVIS1 1
//#define TRAVISKi67 1
//#define TRAVIS_02 1
// 3D
#define TOOTH 1
//#define DEEP_SYNTH_3D 1
//#define SPHERES 1
#if ZEBRA
    //Zebra
    std::string input_image_filename("d:/ImagingData/bhat/2d/zebra/image.bmp");
    std::string input_mask_filename("d:/ImagingData/bhat/2d/zebra/mask2.bmp");
#elif CHEETAH
    //Cheetah
    std::string input_image_filename("d:/ImagingData/bhat/2d/cheetah/image.bmp");
    std::string input_mask_filename("d:/ImagingData/bhat/2d/cheetah/mask.bmp");
#elif CAT
    //CAT
    std::string input_image_filename("d:/ImagingData/bhat/2d/cat/image.bmp");
    std::string input_mask_filename("d:/ImagingData/bhat/2d/cat/mask.bmp");
#elif DEEP_SYNTH_2D
    std::string input_image_filename("d:/ImagingData/bhat/0017.bmp");
    std::string input_mask_filename("d:/ImagingData/bhat/0017_mask.bmp");
#elif SAAD1
    std::string input_image_filename("d:/ImagingData/bhat/2d/Saad/H15-615Hodgkin.png");
    std::string input_mask_filename("d:/ImagingData/bhat/2d/Saad/Mask2_H15-615Hodgkin.png");
#elif TRAVIS1
    std::string input_image_filename("d:/ImagingData/bhat/2d/Saad/travis_Point1.jpg");
    std::string input_mask_filename("d:/ImagingData/bhat/2d/Saad/Mask1_travis_Point1.jpg");
#elif TRAVISKi67
    std::string input_image_filename("d:/ImagingData/bhat/2d/Saad/Ki67/travis_ki67_tumor_b.png");
    std::string input_mask_filename("d:/ImagingData/bhat/2d/Saad/Ki67/Mask_travis_ki67_tumor_b.png");
#elif TRAVIS_02
    std::string input_image_filename("d:/ImagingData/bhat/2d/Saad/02/"\
    "travis_20191203_Breast-CID_FOV16-18_Point1_Overlay_dsDNA-original.jpg");
    std::string input_mask_filename("d:/ImagingData/bhat/2d/Saad/02/mask.jpg");
#elif SPHERES
    std::string input_image_filename("d:/ImagingData/bhat/spheres/Spheres.mhd");
    std::string input_mask_filename("d:/ImagingData/bhat/spheres/seg_out.mhd");
#elif TOOTH
    std::string input_image_filename("f:/ImagingData/bhat/tooth/tooth.mhd");
    std::string input_mask_filename("");
#elif DEEP_SYNTH_3D
    std::string input_image_filename("d:/ImagingData/bhat/test1/data");
    std::string input_mask_filename( "d:/ImagingData/bhat/test1/mask");
#endif

vtkSmartPointer<vtkImageData> CreateROIMask(vtkImageData* inputImage, int shape)
{
    vtkNew<vtkROIStencilSource> stencil_source;

    //Get and shrink bounds by 20 percent along each axis.
    double stencil_bounds[6];
    inputImage->GetBounds(stencil_bounds);
    auto dx = 0.4*(stencil_bounds[1] - stencil_bounds[0]);
    auto dy = 0.4*(stencil_bounds[3] - stencil_bounds[2]);
    auto dz = 0.4*(stencil_bounds[5] - stencil_bounds[4]);
    stencil_bounds[0] += dx; stencil_bounds[1] -= dx;
    stencil_bounds[2] += dy; stencil_bounds[3] -= dy;
    stencil_bounds[4] += dz; stencil_bounds[5] -= dz;

    stencil_source->SetBounds(stencil_bounds);
    stencil_source->SetShape(shape);
    stencil_source->SetInformationInput(inputImage);
    stencil_source->Update();
    
    // Convert stencil to vtkImage
    vtkNew<vtkImageStencilToImage> conv;
    conv->SetInputData(stencil_source->GetOutput());
    conv->Update();
    return conv->GetOutput();
}

vtkSmartPointer<vtkImageData> ReadMask(const std::string& filename, vtkSmartPointer<vtkImageData> inputImage)
{
    vtkSmartPointer<vtkImageData> ret = nullptr;
    if (!filename.empty())
    {
        ret = ReadImage(filename);
    }

    if (!ret && inputImage)
    {
        ret = CreateROIMask(inputImage, vtkROIStencilSource::BOX);
        //ret = CreateROIMask(inputImage, vtkROIStencilSource::ELLIPSOID);
    }

    return ret;
}


#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"

//void CallMatlab(matlab::engine::MATLABEngine* matlabPtr, std::vector<float>& phi, float dt, int dim[3])
void CallMatlab(matlab::engine::MATLABEngine* matlabPtr, float* phi, size_t phi_size, float dt, int dim[3])
{
    // Create MATLAB data array factory
    matlab::data::ArrayFactory factory;

    matlab::data::ArrayDimensions d{ (size_t)dim[0], (size_t)dim[1], (size_t)dim[2] };
    if (d.back() <= 1)
        d.pop_back();

    matlab::data::TypedArray<float> phi_input = factory.createArray<float>(d, phi, phi + phi_size);

    // Pass vector containing 2 scalar args in vector
    std::vector<matlab::data::Array> args({ phi_input, factory.createScalar<float>(dt) });

    // Call MATLAB function and return result
    matlab::data::TypedArray<float> phi_output = matlabPtr->feval(u"sussman3D", args);
    //std::copy(phi_output.data(), phi_output.data() + phi_output., phi.begin());
    size_t i = 0;
    for (auto iter = phi_output.begin(); iter != phi_output.end(); ++iter)
    {
        phi[i++] = *iter;
    }
}

std::future<vtkSmartPointer<vtkRenderWindow>> bhat_proc()
{
    // Pass vector containing MATLAB data array scalar
    using namespace matlab::engine;
    // Start MATLAB engine synchronously
    std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = startMATLAB();

    vtkSmartPointer<vtkImageData> inputImage = nullptr;
    vtkSmartPointer<vtkImageData> inputMask  = nullptr;
    inputImage = ReadImage(input_image_filename);
    inputMask  = ReadMask(input_mask_filename, inputImage);
    Bhat::FullRegion baseRegion = Bhat::FullRegion(inputImage);

    MacroAssert(inputImage->GetNumberOfScalarComponents() == 1);
    MacroAssert(inputMask->GetNumberOfScalarComponents() == 1);

    // Print Input Information
    double range[2];
    inputImage->GetScalarRange(range);
    std::cout << "input range = " << range[0] << ", " << range[1] << std::endl;

    //Bhat::FullRegion baseRegion(inputBaseRegion);
    size_t baseSize = baseRegion.Size();

    int inputDim[3] = { 0,0,0 };
    inputImage->GetDimensions(inputDim);
    std::cout << "input dimensions = " << inputDim[0] << ", " << inputDim[1] << ", " << inputDim[2] << std::endl;
    double spacing[3] = { 0,0,0 };
    inputImage->GetSpacing(spacing);
    std::cout << "input spacing = " << spacing[0] << ", " << spacing[1] << ", " << spacing[2] << std::endl;
    gParam.SetNarrowBand(gParam.NarrowBand() * double(*std::max_element(spacing, spacing + 3)));

    vtkSmartPointer<vtkImageData> vtk_mask = ConstructFloatImage(inputDim, spacing);
    vtkSmartPointer<vtkImageData> alphaK   = ConstructFloatImage(inputDim, spacing);
    vtkSmartPointer<vtkImageData> phi_image = ConstructFloatImage(inputDim, spacing);
    vtkSmartPointer<vtkImageData> display_phi = ConstructFloatImage(inputDim, spacing);
    vector<float> V(baseSize);
    vector<float> E(baseSize);
    float* phi = (float*)phi_image->GetScalarPointer();

    Bitvector mask(baseSize);
    ConstructMask(inputMask, baseRegion, mask);
    construct_phi_cuda(mask, baseRegion, phi);

    vector<Space::Point> points = GetVoxels(inputImage, baseRegion);

    Space space(points, mask);
    space.BuildKernel();

    std::future<vtkSmartPointer<vtkRenderWindow>> win_future;
    if (inputDim[2] == 1)
        win_future = std::async(std::launch::async, show_progress, inputImage, display_phi);
    else if (inputDim[2] > 1)
        win_future = std::async(std::launch::async, show_progress_3D, inputImage, display_phi);
    else
        MacroFatalError("Wrong input dimensions. Cannot proceed.");

    //getch();
    QElapsedTimer timer;
    timer.start();
    qint64 prevTime = timer.elapsed();
    int steps = 0;
    do
    {
        if (steps % 10 == 0)
        {
            std::memcpy(display_phi->GetScalarPointer(), phi_image->GetScalarPointer(), sizeof(float)*phi_image->GetNumberOfPoints());
            display_phi->Modified();
        }
        //construct_mask_vector(phi, baseRegion, mask);
        //qint64 t1 = timer.elapsed();
        ConstructMask(phi, (size_t)phi_image->GetNumberOfPoints(), mask);

        if (steps % gParam.RecomputePhiIterations() == 0)
        {
            //t1 = timer.elapsed();
#if 0
            construct_phi_cuda(mask, baseRegion, phi);
#else
            CallMatlab(matlabPtr.get(), phi, (size_t)phi_image->GetNumberOfPoints(), 0.5f, inputDim);
            ConstructMask(phi, (size_t)phi_image->GetNumberOfPoints(), mask);
#endif
            //std::cout << "\nconstruct_phi_cuda = " << timer.elapsed() - t1;

            qint64 currentTime = timer.elapsed();
            std::cout << "Iteration time = " << (currentTime - prevTime) / gParam.RecomputePhiIterations() << " ms \n";
            prevTime = currentTime;
        }

        //t1 = timer.elapsed();
        space.UpdateMask(mask);
        space.ComputeP2();
        space.ComputeL();
        space.ComputeV(V, gParam.Del(), gParam.Pi());
        //std::cout << "\nspace.computes = " << timer.elapsed() - t1;

        //t1 = timer.elapsed();
        //UpdatePhiImage(phi, baseRegion, phi_image);
        //std::cout << "\nUpdatePhiImage = " << timer.elapsed() - t1;

        //t1 = timer.elapsed();
        compute_K_div(alphaK, phi_image, gParam.Alpha());
        //std::cout << "\ncompute_K_div= " << timer.elapsed() - t1;

        //t1 = timer.elapsed();
        // Compute alphaK - V
        const float* alphaK_data = (const float*)alphaK->GetScalarPointer();
        for (size_t i = 0; i < baseSize; ++i)
            E[i] = alphaK_data[baseRegion[i]] - std::copysignf(0.25f, V[i]);

        //std::cout << "\nalphaK - V = " << timer.elapsed() - t1;

        if(steps % 20 == 0)
            std::cout << "\nIteration = " << steps << "   \n"; 

        //t1 = timer.elapsed();
        auto E_range = std::minmax_element(E.begin(), E.end());
        double max_e = std::max(abs(*E_range.first), abs(*E_range.second));

        // Compute delta(phi)
        // Update Phi as phi += stepSize * delta_phi * E;
        for(size_t i=0; i < baseSize; ++i)
        {
            double delta_phi = 0.0;
            double current_phi_value = phi[i];
            if (abs(current_phi_value) <= gParam.NarrowBand() || gParam.NarrowBand() < 0)
            {
                //delta_phi = 0.5 / gParam.del * (1.0 + std::cos(gParam.pi * current_phi_value / gParam.del));
                phi[i] += (gParam.StepSize() / (max_e + gParam.Eps())) * E[i];
            }
        }
        //std::cout << "\nphi narrowband update = " << timer.elapsed() - t1 << std::endl;
    } while (++steps < gParam.NumOfIterations());
    _getch();

    int outDim[] = { 128, 128, 128 };
    WriteOutputMask(vtk_mask, outDim);
    return win_future;
}


/*
std::future<vtkSmartPointer<vtkRenderWindow>> bhat_gpu()
{
    int count = 0;
    cudaGetDeviceCount(&count);
    std::cout << "Cuda device count = " << count << std::endl;

    vtkSmartPointer<vtkImageData> inputImage = nullptr;
    vtkSmartPointer<vtkImageData> inputMask  = nullptr;
    inputImage = ReadImage(input_image_filename, true);
    inputImage = AddNoise(inputImage, 30);
    inputMask  = ReadMask(input_mask_filename, inputImage);
    Bhat::FullRegion baseRegion = Bhat::FullRegion(inputImage);

    MacroAssert(inputImage->GetNumberOfScalarComponents() == 1);
    MacroAssert(inputMask->GetNumberOfScalarComponents() == 1);

    // Print Input Information
    double range[2];
    inputImage->GetScalarRange(range);
    std::cout << "input range = " << range[0] << ", " << range[1] << std::endl;

    //Bhat::FullRegion baseRegion(inputBaseRegion);
    size_t baseSize = baseRegion.Size();

    int inputDim[3] = { 0,0,0 };
    inputImage->GetDimensions(inputDim);
    std::cout << "input dimensions = " << inputDim[0] << ", " << inputDim[1] << ", " << inputDim[2] << std::endl;
    double spacing[3] = { 0,0,0 };
    inputImage->GetSpacing(spacing);
    std::cout << "input spacing = " << spacing[0] << ", " << spacing[1] << ", " << spacing[2] << std::endl;
    gParam.SetNarrowBand(gParam.NarrowBand() * double(*std::max_element(spacing, spacing + 3)));

    vtkSmartPointer<vtkImageData> vtk_mask = ConstructFloatImage(inputDim, spacing);
    vtkSmartPointer<vtkImageData> alphaK   = ConstructFloatImage(inputDim, spacing);
    vtkSmartPointer<vtkImageData> phi_image = ConstructFloatImage(inputDim, spacing);
    vtkSmartPointer<vtkImageData> display_phi = ConstructFloatImage(inputDim, spacing);
    vector<float> E(baseSize);
    vector<float> phi(baseSize);

    Bitvector mask(baseSize);
    ConstructMask(inputMask, baseRegion, mask);
    construct_phi_cuda(mask, baseRegion, phi.data());
    vector<PrecType> points = GetCudaVoxels(inputImage, baseRegion);

    vector<bool> btmp(mask.Size());
    for (size_t i = 0; i < mask.Size(); ++i)
        btmp[i] = mask.Get(i);

    SpaceCuda space(points, btmp, inputDim);
    space.SetPhi(phi);
    space.BuildKernel();

    std::future<vtkSmartPointer<vtkRenderWindow>> win_future;
    if (inputDim[2] == 1)
        win_future = std::async(std::launch::async, show_progress, inputImage, display_phi);
    else if (inputDim[2] > 1)
        win_future = std::async(std::launch::async, show_progress_3D, inputImage, display_phi);
    else
        MacroFatalError("Wrong input dimensions. Cannot proceed.");

    QElapsedTimer timer;
    timer.start();
    qint64 startTime = timer.elapsed();
    qint64 prevTime = 0, kdivTime = 0;
    int steps = 0;
    do
    {
        if (steps % 100 == 0)
        {
            space.GetPhi(phi);
            std::memcpy(display_phi->GetScalarPointer(), phi.data(), sizeof(float)*phi.size());
            //phi_image->Modified();
            //std::memcpy(display_phi->GetScalarPointer(), phi_image->GetScalarPointer(), sizeof(float)*phi_image->GetNumberOfPoints());
            display_phi->Modified();
        }
        space.UpdateMaskByPhi();

        if (steps % gParam.RecomputePhiIterations() == 0)
        {
            std::vector<bool> tmp_mask;
            space.GetMask(tmp_mask);
            mask.ClearBits();
            for (size_t i = 0; i < tmp_mask.size(); ++i)
                if (tmp_mask[i]) mask.Set(i);

            qint64 t = timer.elapsed();

            construct_phi_cuda(mask, baseRegion, phi.data());
            std::cout << "Phi calculation time = " << timer.elapsed() - t << "\n";

            space.SetPhi(phi);

            qint64 currentTime = timer.elapsed();
            std::cout << "Iteration time = " << (currentTime - prevTime) / gParam.RecomputePhiIterations() << " ms, ";
            std::cout << "kdiv time = " << kdivTime / 1000 << " s, ";
            std::cout << "Total time = " << (currentTime - startTime) / 1000 << " s \n";
            prevTime = currentTime;
            //kdivTime = 0;
        }

        space.ComputeP2();
        space.ComputeL();
        space.ComputeV(gParam.Del(), gParam.Pi());

#if 1
        space.ComputeDivergence(gParam.Alpha(), spacing, gParam.NarrowBand());
        space.ProcessDivergence(gParam.StepSize(), gParam.NarrowBand());
#else
        auto t = timer.elapsed();
        space.GetPhi(phi);
        std::memcpy(phi_image->GetScalarPointer(), phi.data(), sizeof(float)*phi.size());
        phi_image->Modified();
        compute_K_div(alphaK, phi_image, gParam.Alpha()); 
        kdivTime += timer.elapsed() - t;
        // Compute alphaK - V
        const float* alphaK_data = (const float*)alphaK->GetScalarPointer();
        space.ProcessDivergence(alphaK_data, gParam.StepSize(), gParam.NarrowBand());
#endif
    }
    while (++steps < gParam.NumOfIterations());

    _getch();

    int outDim[] = { 128, 128, 128 };
    WriteOutputMask(vtk_mask, outDim);
    return win_future;
}
*/

/* known to work
DEEP_SYNTH_3D
    gParam.SetParameters(0.000005, 0.5, 1.0, 5.0, 20000, 400, 1); 

SPHERES
    gParam.SetParameters(0.000005, 0.5, 1.0, 3.0, 20000, 200, 1); 

ZEBRA
    gParam.SetParameters(0.00058, 0.5, 1.0, 5.0, 20000, 1000, 1); 
    gParam.SetParameters(0.0005, 0.5, 1.0, 5.0, 20000, 1000, 1); 

*/

int main(int argc, char** argv)
{
    gParam.SetParameters(0.0001, 1.0, 1.0, 3.0, 10000, 100, 1);
#if SAAD1
    gParam.SetParameters(0.0001, 10.0, 1.0, 3.0, 20000, 500, 2); // works really well on H15-615
#elif TRAVIS1
    gParam.SetParameters(0.0001, 10.0, 1.0, 3.0, 20000, 500, 4); // works really well on Travis_1
#elif TRAVISKi67
    gParam.SetParameters(0.0001, 10.0, 1.0, 3.0, 20000, 500, 4);
#elif TRAVIS_02
    gParam.SetParameters(0.0001, 10.0, 1.0, 3.0, 20000, 500, 4); 
#elif DEEP_SYNTH_3D
    gParam.SetParameters(0.0005, 0.5, 1.0, 5.0, 20000, 400, 1); 
#elif SPHERES
    gParam.SetParameters(0.000005, 0.2, 1.0, 5.0, 10000, 400, 1); 
#elif TOOTH
    //gParam.SetParameters(0.000005, 0.5, 1.0, 3.0, 20000, 10, 1); 
    gParam.SetParameters(0.2, 0.1, 2.0, 2.0, 20000, 10, 1); 
#elif ZEBRA
    gParam.SetParameters(0.0005, 0.5, 1.0, 5.0, 20000, 500, 1); 
#elif CHEETAH
    gParam.SetParameters(0.00005, 0.1, 1.0, 3.0, 20000, 200, 1); 
#endif
    QApplication app(argc, argv);

    auto win = bhat_proc();
    //input_mask_filename.clear();
    //auto win = bhat_gpu();
    //MainWindow mainWin;
    //mainWin.resize(1000, 1000);
    //mainWin.show();

    return app.exec();
}

