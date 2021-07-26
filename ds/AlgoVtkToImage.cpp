#include <vtkImageData.h>
#include "core/macros.h"
#include "AlgoVtkToImage.h"
#include "ds/Grid.h"
#include "ds/Image.h"
//#include "graphseg_globals.h"

using sjDS::Grid;
using sjDS::Image;
using hseg::AlgoVtkToImage;

AlgoVtkToImage::~AlgoVtkToImage()
{
}

AlgoVtkToImage::AlgoVtkToImage(vtkSmartPointer<vtkImageData>& input) : m_InputImage(input)
{
}

int AlgoVtkToImage::input_validation() const
{
    if (m_InputImage.GetPointer() == nullptr)
        return 0;

    int t = m_InputImage->GetScalarType();
    if (t != VTK_UNSIGNED_INT && t != VTK_UNSIGNED_SHORT && t != VTK_UNSIGNED_CHAR )
    {
        MacroWarning("vtkImageData's scalar type should be 'unsigned: integer, short, or char' :" << t);
        return 0;
    }

    return 1;
}

int AlgoVtkToImage::primary_process()
{
    vtkIdType array_size = 0; 
    int int_dim[3] = {0,0,0};
    double spacing[3] = { 0,0,0 }, bounds[6] = { 0,0,0,0,0,0 };
    array_size = m_InputImage->GetNumberOfPoints();
    m_InputImage->GetDimensions(int_dim);
    //m_InputImage->GetOrigin(origin);
    m_InputImage->GetBounds(bounds);
    double origin[3] = { bounds[0], bounds[2], bounds[4] };
    m_InputImage->GetSpacing(spacing);

    MacroAssert(int_dim[0] >= 0 && int_dim[1] >= 0 && int_dim[2] >= 0);
    size_t dim[3] = {(size_t)int_dim[0], (size_t)int_dim[1], (size_t)int_dim[2]};
    sjDS::Grid grid(dim);
    grid.SetOrigin(origin);
    grid.SetSpacing(spacing);

    const void* scalar_ptr = m_InputImage->GetScalarPointer();
    NullCheck(scalar_ptr, 0);
    int scalar_type = m_InputImage->GetScalarType();
    size_t size = (size_t)m_InputImage->GetNumberOfPoints();
    std::vector<uint> ret;

    switch (scalar_type)
    {
    case VTK_UNSIGNED_INT:
        m_OutputImage = sjDS::Image((const type_uint*)scalar_ptr, &grid);
        break;
    case VTK_UNSIGNED_SHORT:
        ret = _ConvertToUnsignedInt((ushort*)scalar_ptr, size);
        m_OutputImage = sjDS::Image(ret.data(), &grid);
        break;
    case VTK_UNSIGNED_CHAR:
        ret = _ConvertToUnsignedInt((uchar*)scalar_ptr, size);
        m_OutputImage = sjDS::Image(ret.data(), &grid);
        break;
    default:
        MacroWarning("Unhandled data type: " << scalar_type);
        return 0;
    }

    return 1;
    // TODO: why the offset?
    //m_OutputImage.SetOrigin( origin[0]+0.5, origin[1]+0.5, origin[2]+0.5 );
    //m_OutputImage.SetExtent( 0, int(dim[0])-1, 0, int(dim[1])-1, 0, int(dim[2])-1 );
    //m_OutputImage.AllocateScalars( VTK_UNSIGNED_INT, 1 );
}

int AlgoVtkToImage::post_processing()
{
    return 1;
}

std::vector<uint> AlgoVtkToImage::_ConvertToUnsignedInt(ushort* ptr, size_t size)
{
    std::vector<uint> ret(size);
    for (size_t i = 0; i < size; ++i)
    {
        ret[i] = (uint)ptr[i];
    }

    return ret;
}

std::vector<uint> AlgoVtkToImage::_ConvertToUnsignedInt(uchar* ptr, size_t size)
{
    std::vector<uint> ret(size);
    for (size_t i = 0; i < size; ++i)
    {
        ret[i] = (uchar)ptr[i];
    }

    return ret;
}

sjDS::Image AlgoVtkToImage::Convert(vtkSmartPointer<vtkImageData> vtk_image, bool usePassedOffset, uint passedOffset)
{
    double range[2];
    vtk_image->GetScalarRange(range);
    vtkSmartPointer<vtkImageData> tmp = vtk_image;
    //if (range[0] < 0 || usePassedOffset)
    if( tmp->GetScalarType() != VTK_UNSIGNED_CHAR &&
        tmp->GetScalarType() != VTK_UNSIGNED_SHORT &&
        tmp->GetScalarType() != VTK_UNSIGNED_INT )
    {
        uint offset = (uint)fabs(std::floor(range[0]));
        if (usePassedOffset)
        {
            offset = passedOffset;
        }
        vtkNew<vtkImageData> uint_input;
        uint_input->SetDimensions(vtk_image->GetDimensions());
        double bounds[6];
        vtk_image->GetBounds(bounds);
        double origin[3] = { bounds[0], bounds[2], bounds[4] };
        uint_input->SetOrigin(origin);
        uint_input->SetSpacing(vtk_image->GetSpacing());
        uint_input->AllocateScalars(VTK_UNSIGNED_INT, 1);
        const void* input_ptr = vtk_image->GetScalarPointer();
        uint* uint_ptr  = (uint*)uint_input->GetScalarPointer();
        vtkIdType arraySize = vtk_image->GetNumberOfPoints();
        int scalar_type = vtk_image->GetScalarType();
        const char* scalar_type_name = vtk_image->GetScalarTypeAsString();
        for (vtkIdType p = 0; p < arraySize; ++p)
        {
            switch (scalar_type)
            {
            case VTK_INT:
                uint_ptr[p] = uint(((int*)input_ptr)[p]) + offset;
                break;
            case VTK_UNSIGNED_INT:
                uint_ptr[p] = uint(((uint*)input_ptr)[p]) + offset;
                break;
            case VTK_SHORT:
                uint_ptr[p] = uint(((short*)input_ptr)[p]) + offset;
                break;
            case VTK_UNSIGNED_SHORT:
                uint_ptr[p] = uint(((ushort*)input_ptr)[p]) + offset;
                break;
            case VTK_CHAR:
                uint_ptr[p] = uint(((char*)input_ptr)[p]) + offset;
                break;
            case VTK_UNSIGNED_CHAR:
                uint_ptr[p] = uint(((uchar*)input_ptr)[p]) + offset;
                break;
            case VTK_FLOAT:
                uint_ptr[p] = uint(((float*)input_ptr)[p]) + offset;
                break;
            default:
                MacroWarning("Unhandled data type: " << scalar_type_name);
                break;
            }
        }
        tmp = uint_input;
    }
    hseg::AlgoVtkToImage converter(tmp);
    converter.Run();
    return converter.GetOutput();
}
