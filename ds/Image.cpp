#include <set>
#include <vector>

#include <vtkImageData.h>
#include <QImage>
#include <QFile>

#include "ds/VoxelRow.h"
#include "Image.h"
#include "ImageSegmentation.h"
#include "Grid.h"
#include "core/macros.h"

#define HSEG_TYPE_UINT_MAX UINT32_MAX
#define HSEG_TYPE_UINT_MIN 0

namespace sjDS{

void Image::init()
{
    m_grid = NULL;
    m_griddata = NULL;
    m_range[0] = HSEG_TYPE_UINT_MAX;
    m_range[1] = HSEG_TYPE_UINT_MIN;
}

int Image::allocate_data_array()
{
    // Check that grid object is constructed.
    NullCheck( m_grid, 0 );

    // Delete the array if previously allocated.
    //MacroFree( m_griddata );
    MacroDeleteArray( m_griddata );

    // Data array size is determined by the grid.
    size_t array_sz = m_grid->GetArraySize();
    m_griddata = new type_uint[array_sz];
    // Intialize to zeroes
    memset( m_griddata, 0, sizeof(type_uint)*array_sz );

    // Return success.
    return 1;
}

Image::~Image() noexcept
{
    MacroDelete(m_grid);
    //MacroFree(m_griddata);
    MacroDeleteArray(m_griddata);
}

Image::Image()
{
    init();
    Grid* grid = new Grid();

    size_t dim[3] = {0,0,0};
    grid->SetDimensions(dim);
    m_grid = grid;
}

Image::Image( const Image& img )
{
    init();
    m_grid = new Grid(*(img.m_grid));
    allocate_data_array();
    memcpy( m_griddata, img.m_griddata, img.GetArraySize()*sizeof(type_uint) );
}

Image::Image(vtkImageData* img)
{
    init();
    NullCheckVoid(img);

    // Construct the Grid object.
    Grid* grid = new Grid();

    int vtk_dim[3] = {0,0,0};
    img->GetDimensions(vtk_dim);

    size_t dim[3] = {0,0,0};
    for(int i=0; i<3; ++i)
        dim[i] = (size_t)vtk_dim[i];

    //TEMP: overwrite
    //dim[2] = 2;

    // Assign dimensions, origin and grid point spacing.
    grid->SetDimensions( dim );
    grid->SetOrigin( img->GetOrigin() );
    grid->SetSpacing( img->GetSpacing() );

    // Grid object cannot be modified after this:
    m_grid = (Grid*)grid;

    // Copy image data array
    double range[2]={0,0};
    img->GetScalarRange(range);
    union unionptr
    {
        char* char_ptr;
        unsigned char* uchar_ptr;
        short* short_ptr;
        unsigned short* ushort_ptr;
        int* int_ptr;
        float* float_ptr;
        unsigned int* uint_ptr = nullptr;
    } uptr;

    int scalarType = img->GetScalarType();
    if( scalarType == VTK_SHORT )
        uptr.short_ptr = (short*)img->GetScalarPointer();
    else if( scalarType == VTK_UNSIGNED_SHORT )
        uptr.ushort_ptr = (unsigned short*)img->GetScalarPointer();
    else if( scalarType == VTK_UNSIGNED_INT )
        uptr.uint_ptr = (unsigned int*)img->GetScalarPointer();
    else if( scalarType == VTK_INT )
        uptr.int_ptr = (int*)img->GetScalarPointer();
    else if( scalarType == VTK_UNSIGNED_CHAR )
        uptr.uchar_ptr = (unsigned char*)img->GetScalarPointer();
    else if( scalarType == VTK_CHAR )
        uptr.char_ptr = (char*)img->GetScalarPointer();
    else if( scalarType == VTK_FLOAT )
        uptr.float_ptr = (float*)img->GetScalarPointer();
    else
    {
        MacroWarning("VTK_SCALAR_TYPE not supported.");
        return;
    }

    // Allocate memory for data array
    allocate_data_array();

    for(size_t i=0; i < m_grid->GetArraySize(); ++i )
    {
        double value = 0;
        switch (scalarType) {
        case VTK_SHORT:
            value =  (double)(uptr.short_ptr[i] - ((short)range[0]));
            break;
        case VTK_UNSIGNED_SHORT:
            value =  (double)(uptr.ushort_ptr[i] - ((unsigned short)range[0]));
            break;
        case VTK_UNSIGNED_INT:
            value =  (double)(uptr.uint_ptr[i] - ((unsigned int)range[0]));
            break;
        case VTK_INT:
            value =  (double)(uptr.int_ptr[i] - ((int)range[0]));
            break;
        case VTK_UNSIGNED_CHAR:
            value =  (double)(uptr.uchar_ptr[i] - ((unsigned char)range[0]));
            break;
        case VTK_CHAR:
            value =  (double)(uptr.char_ptr[i] - ((char)range[0]));
            break;
        case VTK_FLOAT:
            value =  (double)(uptr.float_ptr[i] - ((float)range[0]));
            break;
        default:
            MacroWarning("Type not supported.");
            return;
        }

        if( value < 0)
        {
            value = 0;
            MacroWarning("Negative voxel value handled. Wrong calculated scalar range.");
        }

        m_griddata[i] = static_cast<type_uint>(value);
    }
}

Image::Image(const QImage* img)
{
    init();
    NullCheckVoid(img);

    int width  = img->width();
    int height = img->height();

    // Construct the Grid object.
    Grid* grid = new Grid();

    double origin[3]  = {0,0,0};
    double spacing[3] = {1,1,1};
    size_t dim[3] = { size_t(width), size_t(height), 1 };

    // Assign dimensions, origin and grid point spacing.
    grid->SetOrigin( origin );
    grid->SetSpacing( spacing );
    grid->SetDimensions( dim );

    // Grid object cannot be modified after this:
    m_grid = (Grid*)grid;

    // Allocate memory for data array
    allocate_data_array();

    int i=0;
    for (int h=0; h < height; ++h)
    {
        for (int w=0; w < width; ++w)
        {
            QRgb color = img->pixel( w, h );

            int red   = qRed   (color);
            int green = qGreen (color);
            int blue  = qBlue  (color);

            // convert to grey scale of range 0 to 65535 (16-bit)
            // we ignore alpha channel
            //int grey = 8*(11*red + 16*green + 5*blue);
            int grey = (11*red + 16*green + 5*blue)/32;

            if( grey < 0 )
            {
                grey = 0;
                MacroWarning("Handled negative grey value.");
            }

            MacroAssert( i < width*height );
            m_griddata[i++] = (type_uint)grey;
        }
    }
}

Image::Image(size_t x, size_t y, size_t z)
{
    init();
    size_t dim[3] = { x,y,z };
    *this = Image(dim);
}

Image::Image(const size_t dim[])
{
    init();

    if( !dim )
    {
        MacroWarning("Data array cannot be NULL.");
        return;
    }

    // Set grid member pointer.
    Grid* tmp_grid = new Grid();
    tmp_grid->SetDimensions(dim);
    m_grid = tmp_grid;

    // Allocate memory for the data array.
    allocate_data_array();
}

Image::Image(const Grid* grid)
{
    init();

    if( !grid )
    {
        MacroWarning("Grid object cannot be NULL.");
        return;
    }

    // Set grid member pointer.
    m_grid = new Grid(*grid);
    // Allocate memory for the data array.
    allocate_data_array();
}

Image::Image(const type_uint* data, const Grid* grid)
{
    init();

    if( !data )
    {
        MacroWarning("Data array cannot be NULL.");
        return;
    }
    if( !grid )
    {
        MacroWarning("Grid object cannot be NULL.");
        return;
    }

    // Set grid member pointer.
    m_grid = new Grid(*grid);
    // Allocate memory for the data array.
    allocate_data_array();

    // Copy data from the provided array to the member array.
    size_t sz = m_grid->GetArraySize();
    memcpy( m_griddata, data, sizeof(type_uint)*sz );
}

Image::Image(const type_uint* data, const size_t dim[3])
{
    init();

    if( !data )
    {
        MacroWarning("Data array cannot be NULL.");
        return;
    }

    // Set grid member pointer.
    Grid* tmp_grid = new Grid();
    tmp_grid->SetDimensions(dim);
    m_grid = tmp_grid;

    // Allocate memory for the data array.
    allocate_data_array();

    // Copy data from the provided array to the member array.
    size_t sz = m_grid->GetArraySize();
    memcpy( m_griddata, data, sizeof(type_uint)*sz );
}

Image::Image(QDataStream& in_stream)
{
    init();
    if( !read( in_stream ) )
        DSObject::SetInvalidConstruction();
}

// Copy Assignment Operator
Image& Image::operator=(const Image& A )
{
    /*
    MacroDelete(m_grid);
    MacroDeleteArray(m_griddata);
    init();

    m_grid = new Grid( *(A.m_grid) );
    allocate_data_array();

    size_t buffer_sz = m_grid->GetArraySize() * sizeof(type_uint);
    memcpy( m_griddata, A.m_griddata, buffer_sz );
    */
    Image tmp(A);           // re-use copy-constructor
    *this = std::move(tmp); // re-use move-assignment
    return *this;
}

// Move Constructor
Image::Image(Image&& img) noexcept
{
    init();
    m_grid = img.m_grid;
    img.m_grid = nullptr;
    m_griddata = img.m_griddata;
    img.m_griddata = nullptr;
    m_range[0] = img.m_range[0];
    m_range[1] = img.m_range[1];
}

// Move Assignment Operator
Image& Image::operator=(Image&& A) noexcept
{
    MacroDelete(m_grid);
    MacroDeleteArray(m_griddata);
    init();
    m_grid = A.m_grid;
    A.m_grid = nullptr;
    m_griddata = A.m_griddata;
    A.m_griddata = nullptr;
    m_range[0] = A.m_range[0];
    m_range[1] = A.m_range[1];

    return *this;
}

type_uint Image::GetVoxel(type_uid voxel_id) const
{
    /*
    NullCheck(m_grid, 0);
    NullCheck(m_griddata, 0);
    size_t array_sz = m_grid->GetArraySize();
    if( voxel_id < array_sz )
        return m_griddata[voxel_id];
    else
        return 0;
        */
    return m_griddata[voxel_id];
}
type_uint Image::GetVoxel(size_t ijk[3]) const
{
    uint voxel_id = m_grid->CalcVoxelID(ijk);
    return m_griddata[voxel_id];
}

int Image::SetVoxel(type_uid voxel_id, type_uint value)
{
    /*
    NullCheck(m_grid, 0);
    NullCheck(m_griddata, 0);

    size_t array_sz = m_grid->GetArraySize();

    if( voxel_id < array_sz )
    {
        m_griddata[voxel_id] = value;
        return 1;
    }
    else
        return 0;
        */
    m_griddata[voxel_id] = value;
    return 1;
}

int Image::SetVoxel(size_t i, size_t j, size_t k, type_uint value)
{
    //NullCheck(m_grid, 0);
    uint voxId = m_grid->CalcVoxelID(i,j,k);
    return SetVoxel(voxId, value);
}

int Image::SetVoxel(size_t ijk[3], type_uint value)
{
    //NullCheck(m_grid, 0);
    uint voxId = m_grid->CalcVoxelID(ijk);
    return SetVoxel(voxId, value);
}

int Image::SetVoxelRow(const VoxelRow& row, type_uint value)
{
    MacroAssert( row.Start() < m_grid->GetArraySize() && row.End() < m_grid->GetArraySize() );
    //memset( &(m_griddata[row.Start()]), value, sizeof(uint)*(size_t)row.Length() );
    for( uint i= row.Start(); !row.atEnd(i); ++i )
        m_griddata[i] = value;

    return 1;
}

int Image::GetScalarRange(type_uint range[2])
{
    NullCheck(range, 0);
    int ret = 0;
    if( !read_scalar_range(range) )
    {
        calculate_scalar_range(range);
        m_range[0] = range[0];
        m_range[1] = range[1];
        ret = 1;
    }

    return ret;
}

int Image::GetScalarRange(type_uint range[2]) const
{
    NullCheck(range, 0);
    int ret = 0;
    if( !read_scalar_range(range) )
    {
        //std::cout << "calculate_scalar_range" << std::endl;
        calculate_scalar_range(range);
        m_range[0] = range[0];
        m_range[1] = range[1];
        ret = 1;
    }

    return ret;
}

int Image::GetScalarRange(type_uint range[2], const std::vector<sjDS::VoxelRow>& region) const
{
    int ret = 1;
    range[0] = UINT_MAX; range[1] = 0;
    for( auto& r : region)
    {
        for( sjDS::voxelNo_t v = r.Start(); !r.atEnd(v); ++v )
        {
            uint value = GetVoxel(v);
            range[0] = value < range[0]? value : range[0];
            range[1] = value > range[1]? value : range[1];
        }
    }

    return ret;
}

bool Image::is2D() const
{
    if( m_grid )
        return m_grid->is2D();
    else
        return false;
}

bool Image::is3D() const
{
    if( m_grid )
        return m_grid->is3D();
    else
        return false;
}

size_t Image::GetArraySize() const
{
    if( m_grid )
        return m_grid->GetArraySize();
    else
        return 0;
}

void Image::GetDimensions( size_t dim[3] ) const
{
    if( m_grid )
        m_grid->GetDimensions( dim );
    else
    {
        dim[0] = 0;
        dim[1] = 0;
        dim[2] = 0;
    }
}

void Image::RecomputeScalarRange() const
{
    m_range[0] = HSEG_TYPE_UINT_MAX;
    m_range[1] = HSEG_TYPE_UINT_MIN;
}

int Image::read_scalar_range(type_uint range[2]) const
{
    int ret = 0;
    range[0] = HSEG_TYPE_UINT_MAX;  // maximum value of data type
    range[1] = HSEG_TYPE_UINT_MIN;  // minimum value of data type

    if( m_range[0] != HSEG_TYPE_UINT_MAX )
    {
        range[0] = m_range[0];
        range[1] = m_range[1];
        ret = 1;
    }

    return ret;
}

int Image::calculate_scalar_range(type_uint range[2]) const
{
    range[0] = HSEG_TYPE_UINT_MAX;  // maximum value of data type
    range[1] = HSEG_TYPE_UINT_MIN;  // minimum value of data type

    size_t sz = m_grid->GetArraySize();

    for(size_t i=0; i < sz; ++i)
    {
        type_uint value = m_griddata[i];

        if( value < range[0])
            range[0] = value;

        if( value > range[1])
            range[1] = value;
    }

    return 1;
}

size_t Image::x() const
{
    NullCheck(m_grid, 0);
    return m_grid->x();
}

size_t Image::y() const
{
    NullCheck(m_grid, 0);
    return m_grid->y();
}

size_t Image::z() const
{
    NullCheck(m_grid, 0);
    return m_grid->z();
}

void Image::GetUniqueValues(std::vector<type_uint>& unique_values ) const
{
    std::set<type_uint> tmp;
    size_t numOfVoxels = GetArraySize();
    for( size_t i=0; i < numOfVoxels; ++i)
    {
        tmp.insert( m_griddata[i] );
    }

    unique_values.clear();
    unique_values.insert( unique_values.end(), tmp.begin(), tmp.end() );
}

void Image::ClearData()
{
    memset( (void*)m_griddata, 0, sizeof(type_uint)*GetArraySize() );
}

int Image::Write(const QString& filename) const
{
    QFile imgFile(filename);
    imgFile.open(QIODevice::WriteOnly);
    QDataStream outStream( &imgFile );
    return Write( outStream );
}

int Image::Write(QDataStream& outStream) const
{
    int ret = m_grid->Write( outStream );

    if( ret )
    {
        size_t sz = m_grid->GetArraySize();

        quint32 uint_var = 0;
        for(size_t i=0; i < sz; ++i)
        {
            uint_var = (quint32)m_griddata[i];
            outStream << uint_var;
        }
    }
    else
        MacroWarning("Failed to write Image.");

    return ret;
}

int Image::read(QDataStream& in_stream)
{
    int ret = 1;
    m_grid = new Grid( in_stream );

    if( m_grid->ValidConstruction() )
    {
        size_t sz = m_grid->GetArraySize();
        //MacroFree( m_griddata );
        MacroDeleteArray( m_griddata );
        allocate_data_array();

        quint32 uint_var = 0;
        for(size_t i=0; i < sz; ++i)
        {
            if( !in_stream.atEnd() )
            {
                in_stream >> uint_var;
                m_griddata[i] = (type_uid)uint_var;
            }
            else
            {
                MacroWarning("Unexpected end of data stream.");
                ret = 0;
            }
        }
    }
    else
        ret = 0;

    return ret;
}

ImageSegmentation* Image::ConvertToImageSegmentation() const
{
    ImageSegmentation* seg = new ImageSegmentation( (const type_uint*)m_griddata, *m_grid );
    return seg;
}

void Image::Threshold(type_uint minimum, type_uint maximum)
{
    size_t sz = GetArraySize();
    type_uint* ptr = GetDataPointer();
    for (size_t i = 0; i < sz; ++i)
    {
        if (ptr[i] < minimum)
            ptr[i] = 0;
        else if (ptr[i] > maximum)
            ptr[i] = 0;
    }
}

void Image::SetOrigin(double x, double y, double z)
{
    m_grid->SetOrigin(x, y, z);
}

void Image::GetBounds(double bounds[6]) const
{
    double origin[3] = { 0,0,0 };
    double spacing[3] = { 0,0,0 };
    size_t dim[3] = { 0,0,0 };
    m_grid->GetOrigin(origin);
    m_grid->GetSpacing(spacing);
    m_grid->GetDimensions(dim);
    bounds[0] = origin[0];
    bounds[1] = origin[0] + (dim[0]-1) * spacing[0];
    bounds[2] = origin[1];
    bounds[3] = origin[1] + (dim[1]-1) * spacing[1];
    bounds[4] = origin[2];
    bounds[5] = origin[2] + (dim[2]-1) * spacing[2];
}

bool Image::isSameDim(const Image&A) const
{
    size_t a_dim[3];
    A.GetDimensions(a_dim);
    size_t this_dim[3];
    this->GetDimensions(this_dim);
    return (a_dim[0] == this_dim[0] && a_dim[1] == this_dim[1] && a_dim[2] == this_dim[2]);
}

}













