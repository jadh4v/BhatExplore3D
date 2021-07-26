#pragma once

//#include <cstring>
#include <QString>
#include <vtkSmartPointer.h>
#include "core/macros.h"

namespace sjDS {
    class Image;
    class Grid;
}

class vtkImageData;

class VolumeReader
{
public:
    enum InputType{ NONE, PVM, RAW, MHA, MHD, BMP, PNG, JPG, GIF, NII, NRRD, TIFF, DICOM_FOLDER, ICS_FOLDER, PNG_FOLDER };
    enum SmoothingType { NO_SMOOTHING = 0, MEDIAN_SMOOTHING, GAUSSIAN_SMOOTHING };

    VolumeReader(const QString& inputFileName);
    VolumeReader(const std::string& inputFileName);
    ~VolumeReader();

    InputType GetImageType();
    static InputType GetImageType(const QString& iFileName);

    int Read();

    /// Construct a sjDS::Image from the output vtkImageData and return it.
    sjDS::Image GetOutput() const;
    /// Return the read volume as a VTK image object.
    vtkSmartPointer<vtkImageData> GetVtkOutput() const;
    void SetSmoothing(enum SmoothingType sm);
    void SetKernelSize(double kernelSize);
    static int ProcessSmoothing(sjDS::Image & img, SmoothingType mode, double kernelSize );
    static int ProcessSmoothing(vtkSmartPointer<vtkImageData>& img, SmoothingType mode, double kernelSize );

private:
    vtkSmartPointer<vtkImageData> _ReadInput2DImage(const QString& filename, const char* ext);
    vtkSmartPointer<vtkImageData> _ReadInputDICOMImage(const QString& iPath);
    vtkSmartPointer<vtkImageData> _ReadInputRAWImage(const QString& iPath);
    vtkSmartPointer<vtkImageData> _ReadInputPVMImage(const QString& iFilename);
    vtkSmartPointer<vtkImageData> _ReadInputNIIImage(const QString& iFilename);
    vtkSmartPointer<vtkImageData> _ReadInputNRRDImage(const QString& iFilename);
    vtkSmartPointer<vtkImageData> _ReadImageSeries(const QString& path);
    vtkSmartPointer<vtkImageData> _ReadICS(const QString& filename);
    vtkSmartPointer<vtkImageData> _ReadInputTIFFImage(const QString& iFilename);

    // DATA-MEMBERS
    /// Input file name to read.
    QString m_inputFileName;
    /// Output image created after reading the file.
    /// This will not be deleted by VolumeReader class.
    vtkSmartPointer<vtkImageData> m_output = nullptr;
    //sjDS::Image* m_output = nullptr;  
    /// Enable / Disable Smoothing
    SmoothingType m_smoothing_mode = NO_SMOOTHING;
    double m_kernel_size = 3;
                                
};
