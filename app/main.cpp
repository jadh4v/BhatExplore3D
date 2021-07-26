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
#include <QSurfaceFormat>
#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>

//VTK
#include <vtkNew.h>
#include <vtkOutlineFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageCast.h>
#include <vtkBMPReader.h>
#include <vtkJPEGReader.h>
#include <vtkPNGReader.h>
#include <vtkCamera.h>
#include <vtkImageExtractComponents.h>
#include <vtkImageResize.h>
#include <vtkImageResample.h>
#include <vtkImplicitModeller.h>
#include <vtkContourFilter.h>
#include <vtkImageActor.h>
#include <vtkImageFlip.h>
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
#include <vtkPNGReader.h>
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
#include "DistField/AlgoCudaDistanceField.h"
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

typedef float PrecType;
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

void PrintScalarType(vtkImageData* img, const char* img_name)
{
    cout << img_name << " scalar type: " << img->GetScalarTypeAsString() << endl;
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

/*
int ICS_to_VTK_datatype(const ics::DataType& type)
{
    switch (type)
    {
    case ics::DataType::UInt8:    return VTK_UNSIGNED_CHAR;  
    case ics::DataType::SInt8:    return VTK_CHAR;
    case ics::DataType::UInt16:   return VTK_UNSIGNED_SHORT;  
    case ics::DataType::SInt16:   return VTK_SHORT; 
    case ics::DataType::UInt32:   return VTK_UNSIGNED_INT; 
    case ics::DataType::SInt32:   return VTK_INT; 
    case ics::DataType::UInt64:   return VTK_UNSIGNED_LONG; 
    case ics::DataType::SInt64:   return VTK_LONG; 
    case ics::DataType::Real32:   return VTK_FLOAT; 
    case ics::DataType::Real64:   return VTK_DOUBLE;
    default:                      return VTK_VOID;
    }
}

vtkSmartPointer<vtkImageData> ReadICS(std::string filename, bool mask = false)
{
    vtkSmartPointer<vtkImageData> ret = nullptr;
    if (QString(filename.data()).endsWith(".ics", Qt::CaseInsensitive))
    {
        try {
            ics::ICS icsfile(filename, "r");
            auto layout = icsfile.GetLayout();
            if (layout.dimensions.size() != 2 && layout.dimensions.size() != 3)
                return ret;

            int dim[3] = {1, 1, 1};
            double origin[3] = { 0,0,0 };
            double spacing[3] = { 1,1,1 };
            for (size_t i = 0; i < layout.dimensions.size(); ++i)
            {
                dim[i]     = (int)(layout.dimensions[i]);
                origin[i]  = icsfile.GetPosition((int)i).origin;
                spacing[i] = icsfile.GetPosition((int)i).scale;
            }

            ret = vtkSmartPointer<vtkImageData>::New();
            ret->SetOrigin(origin);
            ret->SetDimensions(dim);
            ret->SetSpacing(spacing);
            ret->AllocateScalars(VTK_FLOAT, 1);
            float* ptr = (float*)ret->GetScalarPointer();
            std::vector<ushort> out(dim[0] * dim[1] * dim[2]);
            MacroPrint(icsfile.GetDataSize());
            icsfile.GetData(out.data(), icsfile.GetDataSize());
            for (vtkIdType i = 0; i < ret->GetNumberOfPoints(); ++i)
            {
                ptr[i] = vtkVariant(out[i]).ToFloat();
            }
            ret->Modified();
        }
        catch (...)
        {
            MacroMessage("Not an ICS file.");
        }
    }
    return ret;
}
*/

vtkSmartPointer<vtkImageData> ReadDisplayImage(std::string filename)
{
    vtkSmartPointer<vtkImageData> ret = nullptr;
    int fileType = VolumeReader::GetImageType(QString(filename.data()));
    if (fileType == VolumeReader::BMP)
    {
        vtkNew<vtkBMPReader> reader;
        reader->SetFileName(filename.data());
        reader->Update();
        ret = reader->GetErrorCode() == 0 ? reader->GetOutput() : nullptr;
    }
    else if (fileType == VolumeReader::JPG)
    {
        vtkNew<vtkJPEGReader> reader;
        reader->SetFileName(filename.data());
        reader->Update();
        ret = reader->GetErrorCode() == 0 ? reader->GetOutput() : nullptr;
    }
    else if (fileType == VolumeReader::PNG)
    {
        vtkNew<vtkPNGReader> reader;
        reader->SetFileName(filename.data());
        reader->Update();
        ret = reader->GetErrorCode() == 0 ? reader->GetOutput() : nullptr;
    }

    vtkNew<vtkImageCast> cast;
    cast->SetInputData(ret);
    cast->SetOutputScalarTypeToFloat();
    cast->Update();
    return cast->GetOutput();;
}

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

#if 1
    vtkNew<vtkImageMathematics> imageMath;
    imageMath->SetInput1Data(cast1->GetOutput());
    imageMath->SetInput2Data(cast2->GetOutput());
    imageMath->SetOperationToAdd();
    imageMath->Update();
    auto noisy = imageMath->GetOutput();
#elif 0
    //add gradient
    auto noisy = imageMath->GetOutput();
    MacroAssert(noisy->GetScalarType() == VTK_DOUBLE);
    MacroAssert(noisy->GetNumberOfScalarComponents() == 1);
    float* ptr = (float*)noisy->GetScalarPointer();
    int dim[3];
    noisy->GetDimensions(dim);
    for (vtkIdType z = 0; z < dim[2]; ++z)
    for (vtkIdType y = 0; y < dim[1]; ++y)
    for (vtkIdType x = 0; x < dim[0]; ++x)
    {
        float value = noisy->GetScalarComponentAsFloat(x, y, z, 0);
        noisy->SetScalarComponentFromFloat(x, y, z, 0, value + z/10);
    }
    noisy->Modified();
#else
    int dim[3];
    cast1->GetOutput()->GetDimensions(dim);
    vtkImageData* noisy = cast2->GetOutput();
    for (vtkIdType z = 0; z < dim[2]; ++z)
    for (vtkIdType y = 0; y < dim[1]; ++y)
    for (vtkIdType x = 0; x < dim[0]; ++x)
    {
        float value1 = cast1->GetOutput()->GetScalarComponentAsFloat(x, y, z, 0);
        float value2 = noisy->GetScalarComponentAsFloat(x, y, z, 0);
        //noisy->SetScalarComponentFromFloat(x, y, z, 0, value2 + 20.0f*value1/float(z+1.0f));
        noisy->SetScalarComponentFromFloat(x, y, z, 0, value2 + value1 * float(dim[2] - z) / float(dim[2]));
    }
    noisy->Modified();
#endif

    double range[2];
    noisy->GetScalarRange(range);
    vtkNew<vtkImageShiftScale> shift;
    shift->SetInputData(noisy);
    shift->SetShift(range[0] * -1.0);
    shift->SetScale(gParam.ScalarRange()/(range[1] - range[0]));
    shift->SetOutputScalarTypeToFloat();
    shift->Update();

    vtkNew<vtkImageCast> cast3;
    cast3->SetInputData(shift->GetOutput());
    cast3->SetOutputScalarTypeToFloat();
    cast3->Update();

    return cast3->GetOutput();
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

    PrintScalarRange(shift->GetOutput(), "Input Mask");
    return shift->GetOutput();
}

// 2D
//#define TRAVIS_02 1
//#define TRAVISKi67 1
//#define ZEBRA 1
//#define COLOR_ZEBRA 1
//#define CHEETAH 1
//#define TRAVIS1 1
//#define SAAD1 1
// 3D
#define TOOTH 1
//#define DEEP_SYNTH_3D 1
//#define SPHERES 1
//#define VP011 1
//#define NEGHIP 1
//#define BBBC027 1
std::string root_dir("f:/ImagingData");
#if ZEBRA
    //Zebra
    std::string input_image_filename("f:/ImagingData/bhat/2d/zebra/image.bmp");
    std::string input_mask_filename("f:/ImagingData/bhat/2d/zebra/mask2.bmp");
#elif COLOR_ZEBRA
    std::string input_image_filename("f:/ImagingData/bhat/2d/color-zebra.jpg");
    std::string input_mask_filename("");
#elif CHEETAH
    //Cheetah
    std::string input_image_filename("f:/ImagingData/bhat/2d/cheetah/image.bmp");
    std::string input_mask_filename("f:/ImagingData/bhat/2d/cheetah/mask.bmp");
#elif CAT
    //CAT
    std::string input_image_filename("f:/ImagingData/bhat/2d/cat/image.bmp");
    std::string input_mask_filename("f:/ImagingData/bhat/2d/cat/mask.bmp");
#elif DEEP_SYNTH_2D
    std::string input_image_filename("f:/ImagingData/bhat/0017.bmp");
    std::string input_mask_filename("f:/ImagingData/bhat/0017_mask.bmp");
#elif SAAD1
    std::string input_image_filename("f:/ImagingData/bhat/2d/Saad/H15-615Hodgkin.png");
    std::string input_mask_filename("f:/ImagingData/bhat/2d/Saad/Mask2_H15-615Hodgkin.png");
#elif TRAVIS1
    std::string input_image_filename("f:/ImagingData/bhat/2d/Saad/travis_Point1.jpg");
    std::string input_mask_filename("f:/ImagingData/bhat/2d/Saad/Mask1_travis_Point1.jpg");
#elif TRAVISKi67
    std::string input_image_filename("f:/ImagingData/bhat/2d/Saad/Ki67/travis_ki67_tumor_b.png");
    std::string input_mask_filename("f:/ImagingData/bhat/2d/Saad/Ki67/Mask_travis_ki67_tumor_b.png");
#elif TRAVIS_02
    std::string input_image_filename("f:/ImagingData/bhat/2d/Saad/02/"\
    "travis_20191203_Breast-CID_FOV16-18_Point1_Overlay_dsDNA-original.jpg");
    std::string input_mask_filename("f:/ImagingData/bhat/2d/Saad/02/mask.jpg");
#elif SPHERES
    std::string input_image_filename("f:/ImagingData/bhat/spheres/Spheres.mhd");
    std::string input_mask_filename("f:/ImagingData/bhat/spheres/seg_out.mhd");
#elif TOOTH
    std::string input_image_filename(root_dir + std::string("/bhat/tooth/tooth.mhd"));
    std::string input_mask_filename("");
#elif DEEP_SYNTH_3D
    std::string input_image_filename("f:/ImagingData/bhat/test1/data");
    std::string input_mask_filename( "f:/ImagingData/bhat/test1/mask");
#elif VP011
    std::string input_image_filename(root_dir + "/CT/VP/VP011");
    std::string input_mask_filename("");
#elif NEGHIP
    std::string input_image_filename("d:/ImagingData/popular/neghip/neghip.pvm");
    std::string input_mask_filename("");
#elif BBBC027
    std::string input_image_filename("D:/Broad Institute 3D Colon Tissue/BBBC027_highSNR_images_part1/BBBC027_highSNR_images_part1/image-final_0000.ics");
    std::string input_mask_filename("");
#endif

vtkSmartPointer<vtkImageData> CreateROIMask(vtkImageData* inputImage, int shape)
{
    vtkNew<vtkROIStencilSource> stencil_source;

    //Get and shrink bounds by 20 percent along each axis.
    double stencil_bounds[6];
    inputImage->GetBounds(stencil_bounds);
    auto dx = 0.1*(stencil_bounds[1] - stencil_bounds[0]);
    auto dy = 0.1*(stencil_bounds[3] - stencil_bounds[2]);
    auto dz = 0.1*(stencil_bounds[5] - stencil_bounds[4]);
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

void WriteAsTIFF(vtkSmartPointer<vtkImageData> inputImage)
{
    vtkNew<vtkImageData> tmp;
    tmp->DeepCopy(inputImage);
    double spacing[3];
    tmp->GetSpacing(spacing);
    double s = *std::min_element(spacing, spacing + 3);
    s *= 2;
    vtkNew<vtkImageResample> resample;
    resample->SetInputData(tmp);
    resample->SetOutputSpacing(s, s, s);
    resample->Update();
    vtkNew<vtkTIFFWriter> writer;
    writer->SetInputData(resample->GetOutput());
    writer->SetFileName("inputImage_vismale.tif");
    writer->SetCompression(0);
    writer->Write();
}

void WriteAsRAW(vtkSmartPointer<vtkImageData> inputImage)
{
    double spacing[3];
    inputImage->GetSpacing(spacing);
    double s = *std::min_element(spacing, spacing + 3);
    s *= 2;
    vtkNew<vtkImageResample> resample;
    resample->SetInputData(inputImage);
    resample->SetOutputSpacing(s, s, s);
    resample->SetOutputScalarType(VTK_UNSIGNED_CHAR);
    resample->Update();

    vtkNew<vtkImageCast> cast;
    cast->SetInputData(resample->GetOutput());
    cast->SetOutputScalarTypeToUnsignedChar();
    cast->Update();
    vtkNew<vtkMetaImageWriter> writer;
    writer->SetInputData(cast->GetOutput());
    writer->SetFileName("inputImage.mhd");
    writer->SetCompression(0);
    writer->Write();
}

QString map_path = "C:/gitlab/PVR-Features/maps_tooth_16_16_16_24";
void WriteRawMaps(QString& path)
{
    path = QFileDialog::getExistingDirectory(0, "Open DIR","C:/gitlab/");
    QDir dir(path.toLatin1().constData());
    if (!dir.exists())
    {
        MacroWarning("Cannot read path.");
        return;
    }

    QFile out("fmap.raw");
    MacroOpenQFileToWrite(out);

    QStringList ext;
    ext << "*.fmap";
    dir.setNameFilters(ext);
    dir.setFilter(QDir::Files);
    dir.setSorting(QDir::Name);
    auto files = dir.entryList();
    vtkNew<vtkStringArray> fileNames;
    fileNames->SetNumberOfValues(files.size());
    for (int i = 0; i < files.size(); ++i)
    {
        QString value = path + "/" + files[i];
        vtkNew<vtkTIFFReader> reader;
        reader->SetFileName(value.toLatin1().constData());
        reader->Update();
        auto img = reader->GetOutput();
        MacroAssert(img);
        MacroAssert(img->GetScalarType() == VTK_UNSIGNED_CHAR);
        MacroAssert(img->GetScalarPointer());
        out.write((const char*)img->GetScalarPointer(),(size_t)img->GetNumberOfPoints());
    }
    out.close();
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

#include "ProjUtils.h"
#include "utils/utils.h"

vtkSmartPointer<vtkImageData> ConstructTestImage()
{
    int dim[3] = { 5,5,10 };
    double spacing[3] = { 1,1,1 };
    double origin[3] = { 0,0,0 };
    auto inputImage = Utils::ConstructImage(dim, spacing, origin, VTK_FLOAT);
    float* scalars = (float*)inputImage->GetScalarPointer();
    for (vtkIdType id = 0; id < inputImage->GetNumberOfPoints(); ++id)
        scalars[id] = id;

    inputImage->Modified();
    return inputImage;
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

    bool smooth = false;
    inputImage = ReadImage(filename, smooth);
//#define FLIP_INPUT
#ifdef FLIP_INPUT
    vtkNew<vtkImageFlip> flip;
    flip->SetInputData(inputImage);
    flip->SetFilteredAxis(1);
    flip->Update();
    inputImage = flip->GetOutput();
#endif

    if (!inputImage.Get())
        MacroFatalError("Cannot read input datafile.");

    //inputImage = ConstructTestImage();
#if 0
    //WriteAsRAW(inputImage);
    //WriteAsTIFF(inputImage);
    WriteRawMaps(map_path);
    exit(0);
#endif

    if (inputImage->GetDimensions()[2] == 1)
    {
        displayImage = ReadDisplayImage(std::string(filename.toLatin1().constData()));
        inputImage->DeepCopy(displayImage);
    }

    //Utils::WriteVolume(inputImage, "c:/tmp/vismale.mhd",true);
    //inputImage = AddNoise(inputImage, 20);
    //WriteAsRAW(inputImage);
    //exit(0);

    reComputeSpacing(inputImage);
    gParam.SetGlobalRange(inputImage->GetScalarRange());
    UserInterface ui(inputImage, displayImage);
    ui.SetNumberOfBlocks(numOfBlocks);
    ui.show();
    ui.UpdateVisualizer();
    return app.exec();
}
