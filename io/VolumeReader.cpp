// Qt
#include <QDir>
#include <QFile>
#include <QImage>

// VTK
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkDICOMImageReader.h>
#include <vtkImageFlip.h>
#include <vtkMetaImageReader.h>
#include <vtkNIFTIImageReader.h>
#include <vtkNrrdReader.h>
#include <vtkTIFFReader.h>
#include <vtkImageMedian3D.h>
#include <vtkImageGaussianSmooth.h>
#include <vtkMetaImageWriter.h>
#include <vtkStringArray.h>
#include <vtkPNGReader.h>

#include <libics.hpp>

//ITK
#include <itkImage.h>
#include <itkDCMTKImageIO.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesReader.h>
#include <itkNrrdImageIO.h>

// Proj
#include "PVMReader.h"
#include "ds/Grid.h"
#include "ds/Image.h"
#include "ds/AlgoImageToVtk.h"
#include "ds/AlgoVtkToImage.h"
#include "VolumeReader.h"

//#define DISABLE_ITK_VTK_GLUE

#ifndef DISABLE_ITK_VTK_GLUE
#include <itkImageToVTKImageFilter.h>
#endif


using std::cout;
using std::endl;
using sjCore::Algorithm;
using sjDS::Image;
using sjDS::Grid;
using hseg::PVMReader;

VolumeReader::VolumeReader(const QString& inputFileName) : m_inputFileName(inputFileName)
{
}

VolumeReader::VolumeReader(const std::string& inputFileName) : m_inputFileName(inputFileName.c_str())
{
}

VolumeReader::~VolumeReader()
{
}

VolumeReader::InputType VolumeReader::GetImageType()
{
    return GetImageType(m_inputFileName);
}

VolumeReader::InputType VolumeReader::GetImageType( const QString& iFileName )
{
    if( iFileName.endsWith(".jpg", Qt::CaseInsensitive ) )
        return InputType::JPG;

    if( iFileName.endsWith(".bmp", Qt::CaseInsensitive ) )
        return InputType::BMP;

    if( iFileName.endsWith(".gif", Qt::CaseInsensitive ) )
        return InputType::GIF;

    if( iFileName.endsWith(".png", Qt::CaseInsensitive ) )
        return InputType::PNG;

    if( iFileName.endsWith(".pvm", Qt::CaseInsensitive ) )
        return InputType::PVM;

    if( iFileName.endsWith(".mhd", Qt::CaseInsensitive ) )
        return InputType::MHD;

    if( iFileName.endsWith(".mha", Qt::CaseInsensitive ) )
        return InputType::MHA;

    if( iFileName.endsWith(".raw", Qt::CaseInsensitive ) )
        return InputType::RAW;

    if( iFileName.endsWith(".nii", Qt::CaseInsensitive ) )
        return InputType::NII;

    if( iFileName.endsWith(".nrrd", Qt::CaseInsensitive ) )
        return InputType::NRRD;

    if( iFileName.endsWith(".tif", Qt::CaseInsensitive ) || iFileName.endsWith(".tiff", Qt::CaseInsensitive ) )
        return InputType::TIFF;

    QDir dir(iFileName);
    auto files = dir.entryList(QDir::Files);

    if (dir.exists(iFileName) && !files.empty())
    {
        if (files[0].endsWith(".dcm",Qt::CaseInsensitive) || files[0].endsWith(".ima",Qt::CaseInsensitive))
            return InputType::DICOM_FOLDER;
        else if (files[0].endsWith(".png",Qt::CaseInsensitive))
            return InputType::PNG_FOLDER;
        else if (files[0].endsWith(".ics",Qt::CaseInsensitive))
            return InputType::ICS_FOLDER;
    }
    return InputType::NONE;
}

int VolumeReader::Read()
{
    // Check if a filename and path is provided.
    if (m_inputFileName.isEmpty())
    {
        MacroWarning("No filename specified.");
        return 0;
    }

    // Check if file exists.
    if (!QFile::exists(m_inputFileName))
    {
        MacroWarning("File or path does not exist:" << m_inputFileName.toLatin1().constData());
        return 0;
    }

    // Open file to read.
    int type = GetImageType(m_inputFileName);
    m_output = nullptr;

    switch(type)
    {
        case DICOM_FOLDER:  m_output = _ReadInputDICOMImage( m_inputFileName );     break;
        case PNG_FOLDER:    m_output = _ReadImageSeries( m_inputFileName );         break;
        case ICS_FOLDER:    m_output = _ReadICS( m_inputFileName );                 break;
        case PVM:           m_output = _ReadInputPVMImage( m_inputFileName );       break;
        case MHD:
        case MHA:
        case RAW:           m_output = _ReadInputRAWImage( m_inputFileName );       break;
        case BMP:           m_output = _ReadInput2DImage( m_inputFileName, "BMP" ); break;
        case PNG:           m_output = _ReadInput2DImage( m_inputFileName, "PNG" ); break;
        case JPG:           m_output = _ReadInput2DImage( m_inputFileName, "JPG" ); break;
        case GIF:           m_output = _ReadInput2DImage( m_inputFileName, "GIF" ); break;
        case NII:           m_output = _ReadInputNIIImage( m_inputFileName ); break;
        case NRRD:          m_output = _ReadInputNRRDImage( m_inputFileName ); break;
        case TIFF:          m_output = _ReadInputTIFFImage( m_inputFileName ); break;

        default: gLog << "Unknown file type." << endl; return 0;
    }

    if (m_smoothing_mode != NO_SMOOTHING && m_output != nullptr)
    {
        ProcessSmoothing(m_output, m_smoothing_mode, m_kernel_size);
    }

    if (m_output)
        return 1;
    else
        return 0;
}

vtkSmartPointer<vtkImageData> VolumeReader::GetVtkOutput() const
{
    return m_output;
}

Image VolumeReader::GetOutput() const
{
    // Construct a sjDS::Image from the output vtkImageData and return it.
    int dim[3];
    m_output->GetDimensions(dim);
    MacroMessage( "Image Dimensions : " << dim[0] << ", " << dim[1] << ", " << dim[2] );
    double range[2];
    m_output->GetScalarRange(range);
    MacroMessage( "Scalar range : (" << range[0] << ", " << range[1] << ")" );
    const char* type = m_output->GetScalarTypeAsString();
    MacroMessage( "Data type is " << type << "." );
    return hseg::AlgoVtkToImage::Convert(m_output);
    //return Image(m_output);
}

void VolumeReader::SetSmoothing(SmoothingType sm)
{
    m_smoothing_mode = sm;
    if(m_smoothing_mode == GAUSSIAN_SMOOTHING)
        MacroMessage("Using Gaussian smoothing.");
    else if(m_smoothing_mode == MEDIAN_SMOOTHING)
        MacroMessage("Using Median smoothing.");
    else
        MacroMessage("Using No smoothing.");
}

void VolumeReader::SetKernelSize(double kernelSize)
{
    m_kernel_size = kernelSize;
}

vtkSmartPointer<vtkImageData> VolumeReader::_ReadInput2DImage( const QString& filename, const char* ext)
{
    QImage file(filename, ext);
    if( file.isNull() )
    {
        MacroWarning("Unable to read input file.");
        return 0;
    }
    file = file.mirrored(false, true);

    Image img( &file );
    return hseg::AlgoImageToVtk::Convert(img, VTK_UNSIGNED_INT);
}

vtkSmartPointer<vtkImageData> VolumeReader::_ReadInputDICOMImage(const QString& iPath)
{
#if 1
    using PixelType = signed short;
    constexpr unsigned int Dimension = 3;
    using ImageType = itk::Image< PixelType, Dimension >;
    using NamesGeneratorType = itk::GDCMSeriesFileNames;
    NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
    nameGenerator->SetUseSeriesDetails(true);
    nameGenerator->AddSeriesRestriction("0008|0021");
    nameGenerator->SetGlobalWarningDisplay(true);
    nameGenerator->SetDirectory(iPath.toStdString());
    //nameGenerator->Update();
    using SeriesIdContainer = std::vector< std::string >;
    const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
    //std::cout << "nameGenerator size = " << seriesUID.size() << std::endl;

    //MacroAssert(seriesUID.size() == 1);
    if (seriesUID.size() != 1)
        return nullptr;

    auto seriesItr = seriesUID.begin();
    auto seriesEnd = seriesUID.end();
    if (seriesItr != seriesEnd)
    {
        //std::cout << "The directory: ";
        //std::cout << iPath.toStdString() << std::endl;
        //std::cout << "Contains the following DICOM Series: ";
        //std::cout << std::endl;
    }
    else
    {
        std::cout << "No DICOMs in: " << iPath.toStdString() << std::endl;
        return EXIT_SUCCESS;
    }
    while (seriesItr != seriesEnd)
    {
        std::cout << seriesItr->c_str() << std::endl;
        ++seriesItr;
    }
    seriesItr = seriesUID.begin();
    std::string seriesIdentifier;
    //otherwise convert everything
    seriesIdentifier = seriesItr->c_str();
    //seriesItr++;
    //std::cout << "\nReading: ";
    //std::cout << seriesIdentifier << std::endl;
    using FileNamesContainer = std::vector< std::string >;
    FileNamesContainer fileNames = nameGenerator->GetFileNames(seriesIdentifier);
    using ReaderType = itk::ImageSeriesReader< ImageType >;
    ReaderType::Pointer reader = ReaderType::New();
    using ImageIOType = itk::GDCMImageIO;
    ImageIOType::Pointer dicomIO = ImageIOType::New();
    reader->SetImageIO(dicomIO);
    reader->SetFileNames(fileNames);

    vtkNew<vtkImageData> ret;

    #ifndef DISABLE_ITK_VTK_GLUE
        //reader->ForceOrthogonalDirectionOff(); //properly read CTs with gantry tilt
        using ConverterType = itk::ImageToVTKImageFilter<ImageType>;
        ConverterType::Pointer conv = ConverterType::New();
        conv->SetInput(reader->GetOutput());
        conv->Update();
        ret->DeepCopy(conv->GetOutput());
    #else
        MacroWarning("ITK_VTK_Glue module is disabled.");
    #endif
    return ret;
#else
    MacroNewVtkObject( vtkDICOMImageReader, dicom_reader );
    dicom_reader->SetDirectoryName(iPath.toLatin1().constData());
    dicom_reader->Update();
    if( dicom_reader->GetErrorCode() != 0 )
        return nullptr;
    vtkSmartPointer<vtkImageData> vimg = dicom_reader->GetOutput();

    //// FLIP 
    MacroNewVtkObject( vtkImageFlip, flip1 );
    flip1->SetInputData(vimg);
    flip1->SetFilteredAxis(2);
    flip1->Update();
    //MacroNewVtkObject( vtkImageFlip, flip2 );
    //flip2->SetInputData( flip1->GetOutput() );
    //flip2->SetFilteredAxis(2);
    //flip2->Update();
    //vimg = flip2->GetOutput();
    vimg = flip1->GetOutput();
    return vimg;
#endif
}

vtkSmartPointer<vtkImageData> VolumeReader::_ReadInputNIIImage( const QString& iFilename )
{
    MacroNewVtkObject( vtkNIFTIImageReader, reader );
    reader->SetFileName(iFilename.toLatin1().constData());
    reader->Update();
    if( reader->GetErrorCode() != 0 )
        return nullptr;

    vtkImageData* vimg = reader->GetOutput();
    return vimg;
}

vtkSmartPointer<vtkImageData> VolumeReader::_ReadInputNRRDImage(const QString& iFilename)
{
    MacroNewVtkObject(vtkNrrdReader, reader);
    reader->SetFileName(iFilename.toLatin1().constData());
    reader->Update();
    auto ret = reader->GetOutput();
    if (ret && ret->GetNumberOfPoints() != 0)
        return reader->GetOutput();

    auto nrrdImageIO = itk::NrrdImageIO::New();
    nrrdImageIO->SetFileName(iFilename.toStdString());
    nrrdImageIO->ReadImageInformation();
    auto compType = nrrdImageIO->GetComponentType();

    vtkNew<vtkImageData> itk_ret;
    constexpr unsigned int Dimension = 3;
    if (compType == itk::ImageIOBase::INT)
    {
        using PixelType = int;
        using ImageType = itk::Image<PixelType, Dimension>;
        using ReaderType = itk::ImageFileReader<ImageType>;
        ReaderType::Pointer itk_reader = ReaderType::New();
        itk_reader->SetFileName(iFilename.toLatin1().constData());
        itk_reader->Update();

        using ConverterType = itk::ImageToVTKImageFilter<ImageType>;
        ConverterType::Pointer conv = ConverterType::New();
        conv->SetInput(itk_reader->GetOutput());
        conv->Update();
        itk_ret->DeepCopy(conv->GetOutput());
    }
    else if (compType == itk::ImageIOBase::SHORT)
    {
        using PixelType = short;
        using ImageType = itk::Image< PixelType, Dimension >;
        using ReaderType = itk::ImageFileReader< ImageType >;
        ReaderType::Pointer itk_reader = ReaderType::New();
        itk_reader->SetFileName(iFilename.toLatin1().constData());
        itk_reader->Update();

        using ConverterType = itk::ImageToVTKImageFilter<ImageType>;
        ConverterType::Pointer conv = ConverterType::New();
        conv->SetInput(itk_reader->GetOutput());
        conv->Update();
        itk_ret->DeepCopy(conv->GetOutput());
    }
    MacroPrint(itk_ret->GetScalarTypeAsString());
    return itk_ret;
}

vtkSmartPointer<vtkImageData> VolumeReader::_ReadInputRAWImage( const QString& iFilename)
{
    MacroNewVtkObject( vtkMetaImageReader, reader );
    reader->SetFileName(iFilename.toLatin1().constData());
    reader->Update();
    if (reader->GetErrorCode() != 0)
    {
        return nullptr;
    }
    return reader->GetOutput();
}

vtkSmartPointer<vtkImageData> VolumeReader::_ReadInputPVMImage( const QString& iFilename )
{
    MacroWarning("PVMReader not implemented.");
    return 0;
}

int VolumeReader::ProcessSmoothing(Image& img, VolumeReader::SmoothingType mode, double kernelSize)
{
    vtkSmartPointer<vtkImageData> vtk_image = hseg::AlgoImageToVtk::Convert(img, VTK_UNSIGNED_INT);
    ProcessSmoothing(vtk_image, mode, kernelSize);
    img = hseg::AlgoVtkToImage::Convert(vtk_image);
    return 1;
}

int VolumeReader::ProcessSmoothing(vtkSmartPointer<vtkImageData>& img, VolumeReader::SmoothingType mode, double kernelSize)
{
    if(mode == GAUSSIAN_SMOOTHING)
        MacroMessage("Using Gaussian smoothing.");
    else if(mode == MEDIAN_SMOOTHING)
        MacroMessage("Using Median smoothing.");
    else
        MacroMessage("Using No smoothing.");

    int dim[3];
    img->GetDimensions(dim);
    // If smoothing mode is Median Filter:
    if (mode == MEDIAN_SMOOTHING)
    {
        MacroNewVtkObject(vtkImageMedian3D, medianFilter);
        medianFilter->SetInputData(img);

        if(dim[2] <= 1)
            medianFilter->SetKernelSize(kernelSize, kernelSize, 1);
        else if(dim[2] > 1)
            medianFilter->SetKernelSize(kernelSize, kernelSize, kernelSize);
        else
            MacroFatalError("Invalid image dimensions.");

        medianFilter->SetNumberOfThreads(12);
        medianFilter->Update();

        if (medianFilter->GetErrorCode() != 0)
        {
            MacroWarning("Failed to perform median filtering.");
            return 0;
        }

        img = medianFilter->GetOutput();
    }
    // If smoothing mode is Gaussian Filter:
    else if (mode == GAUSSIAN_SMOOTHING)
    {
        MacroNewVtkObject(vtkImageGaussianSmooth, gaussFilter);
        gaussFilter->SetInputData(img);
        if(dim[2] <= 1)
            gaussFilter->SetDimensionality(2);
        else if(dim[2] > 1)
            gaussFilter->SetDimensionality(3);
        else
            MacroFatalError("Invalid image dimensions.");

        gaussFilter->SetRadiusFactor(1);
        gaussFilter->SetStandardDeviation(kernelSize);
        gaussFilter->SetNumberOfThreads(12);
        gaussFilter->Update();

        if (gaussFilter->GetErrorCode() != 0)
        {
            MacroWarning("Failed to perform gaussian filtering.");
            return 0;
        }

        img = gaussFilter->GetOutput();
    }
    return 1;
}

vtkSmartPointer<vtkImageData> VolumeReader::_ReadImageSeries(const QString& path)
{
    QDir dir(path);
    if (!dir.exists())
    {
        MacroWarning("Cannot read path.");
        return nullptr;
    }

    auto files = dir.entryList(QDir::Files);
    vtkNew<vtkStringArray> fileNames;
    fileNames->SetNumberOfValues(files.size());
    for (int i=0; i < files.size(); ++i)
    {
        QString value = path + "/" + files[i];
        fileNames->SetValue(i, value.toLatin1().constData());
    }

    vtkNew<vtkPNGReader> reader;
    reader->SetFileNames(fileNames);
    reader->Update();
    return reader->GetOutput();
}

vtkSmartPointer<vtkImageData> VolumeReader::_ReadICS(const QString& filename)
{
    std::string std_filename(filename.toLatin1().constData());
    vtkSmartPointer<vtkImageData> ret = nullptr;
    if (filename.endsWith(".ics", Qt::CaseInsensitive))
    {
        try {
            ics::ICS icsfile(std_filename, "r");
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

vtkSmartPointer<vtkImageData> VolumeReader::_ReadInputTIFFImage( const QString& iFilename )
{
    MacroNewVtkObject(vtkTIFFReader, reader);
    reader->SetFileName(iFilename.toLatin1().constData());
    reader->Update();
    if( reader->GetErrorCode() != 0 )
        return nullptr;

    return reader->GetOutput();
}
