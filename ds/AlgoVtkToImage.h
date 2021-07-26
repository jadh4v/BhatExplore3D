#pragma once

#include <vtkSmartPointer.h>
#include "core/Algorithm.h"
#include "ds/Image.h"
class vtkImageData;

namespace hseg{

/**
 * @brief The AlgoVtkToImage class
 * Convert a given vtkImageData object into a sjDS::Image object. Data-type should be unsigned int.
 * Provided Image is a regular grid with assigned values as unsigned 32-bit integers.
 */
class AlgoVtkToImage : public sjCore::Algorithm
{
public:
    AlgoVtkToImage(vtkSmartPointer<vtkImageData>& input);
    virtual ~AlgoVtkToImage();
    sjDS::Image GetOutput() const { return m_OutputImage; }
    static sjDS::Image Convert(vtkSmartPointer<vtkImageData> vtk_image, bool usePassedOffset=false, uint passedOffset=0);

private:
    int input_validation() const;
    int primary_process();
    int post_processing();

    std::vector<uint> _ConvertToUnsignedInt(ushort* ptr, size_t size);
    std::vector<uint> _ConvertToUnsignedInt(uchar* ptr, size_t size);

    // Set, not owned.
    vtkSmartPointer<vtkImageData>& m_InputImage;
    // Allocated but not owned.
    sjDS::Image m_OutputImage;
};

}
