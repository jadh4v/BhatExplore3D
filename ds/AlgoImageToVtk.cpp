#include <vtkImageData.h>
#include "core/macros.h"
#include "AlgoImageToVtk.h"
#include "ds/Grid.h"
#include "ds/Image.h"
//#include "graphseg_globals.h"

using sjDS::Grid;
using sjDS::Image;
using hseg::AlgoImageToVtk;

AlgoImageToVtk::~AlgoImageToVtk()
{
}

AlgoImageToVtk::AlgoImageToVtk(const sjDS::Image& input) : m_InputImage(input)
{
}

int AlgoImageToVtk::input_validation() const
{
    if (!m_InputImage.ValidConstruction())
        return 0;

    if (m_OutputDataType != VTK_CHAR
        && m_OutputDataType != VTK_UNSIGNED_CHAR
        && m_OutputDataType != VTK_INT
        && m_OutputDataType != VTK_UNSIGNED_INT
        && m_OutputDataType != VTK_DOUBLE
        )
    {
        MacroWarning("Output data type: " << m_OutputDataType << " is not supported.");
        return 0;
    }

    return 1;
}

int AlgoImageToVtk::primary_process()
{
    const type_uid* data_ptr = m_InputImage.GetDataPointer();
    NullCheck(data_ptr, 0);
    const Grid* grid = m_InputImage.GetGrid();
    NullCheck(grid, 0);
    size_t array_size = 0, dim[3] = {0,0,0};
    double origin[3] = {0,0,0}, spacing[3] = {0,0,0};
    array_size = grid->GetArraySize();
    grid->GetDimensions(dim);
    grid->GetOrigin(origin);
    grid->GetSpacing(spacing);

    m_VtkImage = vtkSmartPointer<vtkImageData>::New();
    // TODO: why the offset?
    //m_VtkImage->SetOrigin( origin[0]+0.5, origin[1]+0.5, origin[2]+0.5 );
    m_VtkImage->SetOrigin(origin);
    m_VtkImage->SetSpacing(spacing);
    m_VtkImage->SetExtent( 0, int(dim[0])-1, 0, int(dim[1])-1, 0, int(dim[2])-1 );
    m_VtkImage->AllocateScalars( m_OutputDataType, 1 );
    vtkIdType numOfPoints = m_VtkImage->GetNumberOfPoints();

    if (m_OutputDataType == VTK_CHAR)
    {
        char* img_ptr = (char*)m_VtkImage->GetScalarPointer();
        for (vtkIdType i = 0; i < numOfPoints; ++i)
        {
            img_ptr[i] = static_cast<char>(m_InputImage[i]);
        }
    }
    else if (m_OutputDataType == VTK_UNSIGNED_CHAR)
    {
        uchar* img_ptr = (uchar*)m_VtkImage->GetScalarPointer();
        for (vtkIdType i = 0; i < numOfPoints; ++i)
        {
            img_ptr[i] = static_cast<uchar>(m_InputImage[i]);
        }
    }
    else if (m_OutputDataType == VTK_SHORT)
    {
        short* img_ptr = (short*)m_VtkImage->GetScalarPointer();
        for (vtkIdType i = 0; i < numOfPoints; ++i)
        {
            img_ptr[i] = static_cast<short>(m_InputImage[i]);
        }
    }
    else if (m_OutputDataType == VTK_UNSIGNED_SHORT)
    {
        ushort* img_ptr = (ushort*)m_VtkImage->GetScalarPointer();
        for (vtkIdType i = 0; i < numOfPoints; ++i)
        {
            img_ptr[i] = static_cast<ushort>(m_InputImage[i]);
        }
    }
    else if (m_OutputDataType == VTK_INT)
    {
        int* img_ptr = (int*)m_VtkImage->GetScalarPointer();
        for (vtkIdType i = 0; i < numOfPoints; ++i)
        {
            img_ptr[i] = static_cast<int>(m_InputImage[i]);
        }
    }
    else if (m_OutputDataType == VTK_UNSIGNED_INT)
    {
        uint* img_ptr = (uint*)m_VtkImage->GetScalarPointer();
        for (vtkIdType i = 0; i < numOfPoints; ++i)
        {
            img_ptr[i] = static_cast<uint>(m_InputImage[i]);
        }
    }
    else if (m_OutputDataType == VTK_DOUBLE)
    {
        double* img_ptr = (double*)m_VtkImage->GetScalarPointer();
        for (vtkIdType i = 0; i < numOfPoints; ++i)
        {
            img_ptr[i] = static_cast<double>(m_InputImage[i]);
        }
    }

    //memcpy( img_ptr, data_ptr, array_size*sizeof(type_uid) );
    return 1;
}

int AlgoImageToVtk::post_processing()
{
    return 1;
}

vtkSmartPointer<vtkImageData> AlgoImageToVtk::Convert(const sjDS::Image& img, int convertedDataType)
{
    hseg::AlgoImageToVtk converter(img);
    converter.SetOutputDataType(convertedDataType);
    converter.Run();
    return converter.GetOutput();
}
