#pragma once

#include <vtkSmartPointer.h>
#include "core/Algorithm.h"
#include "core/macros.h"
class vtkImageData;

namespace sjDS{
class Image;
}

namespace hseg{

/**
 * @brief The AlgoImageToVtk class
 * Convert a given Image object into a vtkImageData using VTK_UNSIGNED_INT.
 * Provided Image is a regular grid with assigned values as unsigned 32-bit integers.
 */
class AlgoImageToVtk : public sjCore::Algorithm
{
public:
    AlgoImageToVtk(const sjDS::Image& seg);
    virtual ~AlgoImageToVtk();
    MacroGetSetMember(int, m_OutputDataType, OutputDataType)
    vtkSmartPointer<vtkImageData> GetOutput() const {return m_VtkImage;}
    static vtkSmartPointer<vtkImageData> Convert(const sjDS::Image& image, int convertedDataType);

private:
    int input_validation() const;
    int primary_process();
    int post_processing();

    // Set, not owned.
    const sjDS::Image& m_InputImage;
    // Allocated but not owned.
    vtkSmartPointer<vtkImageData> m_VtkImage = nullptr;
    int m_OutputDataType = VTK_UNSIGNED_INT;
};

}
