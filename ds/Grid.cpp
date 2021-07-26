#include <cstddef>
#include <QDataStream>
#include <vtkMath.h>
#include <vtkImageData.h>

#include "Grid.h"
#include "GridPoint.h"

namespace sjDS{

//TODO: Should this be uint64_t? This can become an issue for large datasets.
typedef uint32_t type_uid;

void Grid::init()
{
    m_array_size = 0;
    m_dim[0] = m_dim[1] = m_dim[2] = 0;
    m_spacing[0] = m_spacing[1] = m_spacing[2] = 1;
    m_origin[0] = m_origin[1] = m_origin[2] = 0;
}

Grid::Grid()
{
    init();
}

Grid::Grid(const size_t dim[3])
{
    init();
    SetDimensions(dim);
}

Grid::Grid(vtkImageData* img)
{
    init();
    if (img != nullptr)
    {
        this->SetDimensions(img->GetDimensions());
        this->SetSpacing(img->GetSpacing());
        this->SetOrigin(img->GetOrigin());
    }
}

Grid::Grid( QDataStream& in_stream )
{
    init();

    if( !read( in_stream ) )
        DSObject::SetInvalidConstruction();
}

size_t Grid::GetArraySize() const
{
    return m_array_size;
}

void Grid::GetDimensions( size_t dim[3] ) const
{
    dim[0] = m_dim[0];
    dim[1] = m_dim[1];
    dim[2] = m_dim[2];
}
void Grid::GetDimensions( int dim[3] ) const
{
    MacroThreeTimes(dim[i] = static_cast<int>(m_dim[i]));
}

void Grid::SetDimensions(const size_t dim[3])
{
    SetDimensions( dim[0], dim[1], dim[2] );
}

void Grid::SetDimensions(const int dim[3])
{
    SetDimensions( (size_t)dim[0], (size_t)dim[1], (size_t)dim[2] );
}

void Grid::SetDimensions(size_t x, size_t y, size_t z)
{
    m_dim[0] = x;
    m_dim[1] = y;
    m_dim[2] = z;
    recalculate_array_size();
}

void Grid::SetSpacing(double x, double y, double z)
{
    m_spacing[0] = x;
    m_spacing[1] = y;
    m_spacing[2] = z;
}

void Grid::SetSpacing(const double spacing[3])
{
    SetSpacing(spacing[0], spacing[1], spacing[2]);
}

void Grid::SetOrigin(double x, double y, double z)
{
    m_origin[0] = x;
    m_origin[1] = y;
    m_origin[2] = z;
}

void Grid::SetOrigin(const double origin[3])
{
    SetOrigin(origin[0], origin[1], origin[2]);
}

void Grid::GetPoint(size_t ijk[3], double x[3]) const
{
    for( int i=0; i < 3; ++i )
        x[i] = ijk[i]*m_spacing[i] + m_origin[i];
}

void Grid::GetPoint( type_uid voxel_id, double x[3]) const
{
    size_t ijk[3] = {0,0,0};
    if( CalcGridIndex( voxel_id, ijk ) )
    {
        GetPoint( ijk, x );
    }
    else
    {
        x[0] = x[1] = x[2] = 0.0;
    }
}

void Grid::recalculate_array_size()
{
    m_array_size = (m_dim[0] * m_dim[1] * m_dim[2]);
}

type_uid Grid::CalcVoxelID(const size_t* ijk) const
{
    if( !ValidateIndex( ijk) )
        return GridPoint::cInvalidID;

    size_t plane_sz = m_dim[0] * m_dim[1];
    type_uid ret = (type_uid)( plane_sz*ijk[2] + m_dim[0]*ijk[1] + ijk[0] );
    return ret;
}

type_uid Grid::CalcVoxelID( size_t i, size_t j, size_t k ) const
{
    size_t ijk[] = {i, j, k};
    return CalcVoxelID( ijk );
}

int Grid::CalcGridIndex(const type_uid voxel_id, size_t ijk[3] ) const
{
    // NullCheck(ijk, 0);
    ijk[0] = 0;
    ijk[1] = 0;
    ijk[2] = 0;

    if( voxel_id >= this->GetArraySize() )
    {
        //MacroWarning("Invalid voxel ID.");
        ijk[0] = GridPoint::cInvalidID;
        ijk[1] = GridPoint::cInvalidID;
        ijk[2] = GridPoint::cInvalidID;
        return 0; // failure
    }

    if( voxel_id == 0 )
        return 1; // success

    size_t bigVox_id = static_cast<size_t>(voxel_id);

    size_t xy_plane_sz = m_dim[0] * m_dim[1];
    if( xy_plane_sz == 0 )
    {
        MacroWarning("xy-plane array size cannot be zero.");
        return 0; // failure
    }

    // beyond this point, m_dim[0], m_dim[1], xy_plane_sz, voxel_id are all non-zero:

    ijk[2] = bigVox_id / xy_plane_sz;
    size_t rem = bigVox_id % xy_plane_sz;
    if( rem == 0 )
        return 1; // success

    ijk[1] = rem / m_dim[0];
    ijk[0] = rem % m_dim[0];

    return 1; // success
}

int Grid::IJKToNormalized(const size_t ijk[3], double image_normalized_coord[3]) const
{
    for(size_t x=0; x < 3; ++x)
    { 
        image_normalized_coord[x] = m_dim[x] > 1? double(ijk[x]) / double(m_dim[x]-1) : 0;
        image_normalized_coord[x] = vtkMath::ClampValue(image_normalized_coord[x], 0.0, 1.0);
    }

    return 1;
}

int Grid::IJKToNormalized(size_t i, size_t j, size_t k, double image_normalized_coord[3]) const
{
    size_t ijk[] = {i, j, k};
    return IJKToNormalized(ijk, image_normalized_coord);
}

int Grid::NormalizedToIJK(const double image_normalized_coord[3], size_t ijk[3], double cell_normalized_residue[3]) const
{
    for (size_t x = 0; x < 3; ++x)
    {
        if (m_dim[x] > 1)
        { 
            //double normalized_cell_width = 1.0 / (m_dim[x]-1);
            double tmp = image_normalized_coord[x] * (m_dim[x]-1);
            ijk[x] = (size_t)tmp;
            cell_normalized_residue[x] = (tmp - ijk[x]);// / normalized_cell_width;
            ijk[x] = vtkMath::ClampValue(ijk[x], size_t(0), m_dim[x]-1);
        }
        else
        { 
            ijk[x] = 0;
        }
    }
    return 1;
}

bool Grid::ValidateIndex( const size_t ijk[3] ) const
{
    return ValidateIndex( ijk[0], ijk[1], ijk[2] );
}

bool Grid::ValidateIndex( size_t i, size_t j, size_t k ) const
{
    return ( i < m_dim[0] && j < m_dim[1] && k < m_dim[2] );
}

bool Grid::ValidateIndex( const int64_t ijk[3] ) const
{
    if( ( ijk[0] < (int64_t)m_dim[0] && ijk[1] < (int64_t)m_dim[1] && ijk[2] < (int64_t)m_dim[2] )
            && (ijk[0] >= 0 && ijk[1] >=0 && ijk[2] >=0)  )
        return true;
    else
        return false;
}

bool Grid::is2D() const
{
    return ( m_dim[2] == 1 && m_dim[0] > 1 && m_dim[1] > 1 );
}

bool Grid::is3D() const
{
    return ( m_dim[2] > 1 && m_dim[0] > 1 && m_dim[1] > 1 );
}

int Grid::halve()
{
    // correct the dimension to make them even:
    for (size_t i=0; i < 3; ++i)
    {
        if( m_dim[i] % 2 != 0 )
            m_dim[i]++;
    }

    for (size_t i=0; i < 3; ++i)
    {
        // divide dimensions by two:
        m_dim[i] /= 2;
        m_spacing[i] *= 2;
    }

    recalculate_array_size();
    return 1;
}

int Grid::ScaleUpBy2()
{
    for (size_t i=0; i < 3; ++i)
    {
        // divide dimensions by two:
        if(m_dim[i] != 1)
            m_dim[i] *= 2;

        m_spacing[i] /= 2;
    }

    recalculate_array_size();
    return 1;
}

int Grid::Write(QDataStream &outstream) const
{
    // use qt integer type for consistency across platforms and compilers.
    quint64 sizet_variable = 0;

    // write grid dimensions
    for(size_t i=0; i<3; ++i)
    {
        sizet_variable = (quint64) m_dim[i];
        outstream << sizet_variable;
    }

    // write origin coordinates
    for(size_t i=0; i<3; ++i)
    {
        sizet_variable = (quint64) m_origin[i];
        outstream << sizet_variable;
    }

    // write voxel spacing
    for(size_t i=0; i<3; ++i)
    {
        sizet_variable = (quint64) m_spacing[i];
        outstream << sizet_variable;
    }

    return 1;
}

int Grid::read(QDataStream &instream)
{
    int ret = 1;
    // use qt integer type for consistency across platforms and compilers.
    quint64 sizet_variable = 0;

    // read grid dimensions
    for(size_t i=0; i<3; ++i)
    {
        if( !instream.atEnd() )
        {
            instream >> sizet_variable;
            m_dim[i] = (size_t)sizet_variable;
        }
        else
        {
            MacroWarning("Unexpected end of QDataStream.");
            ret = 0;
        }
    }

    // read origin coordinates
    for(size_t i=0; i<3; ++i)
    {
        if( !instream.atEnd() )
        {
            instream >> sizet_variable;
            m_origin[i] = (size_t)sizet_variable;
        }
        else
        {
            MacroWarning("Unexpected end of QDataStream.");
            ret = 0;
        }
    }

    // read voxel spacing
    for(size_t i=0; i<3; ++i)
    {
        if( !instream.atEnd() )
        {
            instream >> sizet_variable;
            m_spacing[i] = (size_t)sizet_variable;
        }
        else
        {
            MacroWarning("Unexpected end of QDataStream.");
            ret = 0;
        }
    }

    recalculate_array_size();

    return ret;
}

bool Grid::SameDimensions(const Grid& A) const
{
    // Check if array sizes are equal.
    if( GetArraySize() != A.GetArraySize() )
        return false;

    // Check if grid dimensions are equal.
    if( x() != A.x() || y() != A.y() || z() != A.z() )
        return false;

    return true;
}

bool Grid::operator==(const Grid& A) const
{
    // Check if dimensions are equal:
    if( !SameDimensions(A) )
        return false;

    for(int i=0;i<3;++i)
    {
        // Check if grid point spacing is equal.
        if( Spacing()[i] != A.Spacing()[i] )
            return false;

        // Check if grid anchor / origin point is equal.
        if( Origin()[i] != A.Origin()[i] )
            return false;
    }

    return true;
}

bool Grid::operator!=(const Grid& A) const
{
    if( this->operator==(A) )
        return false;
    else
        return true;
}

void Grid::GetCenter(double center[3]) const
{
    for (int i = 0; i < 3; ++i)
    {
        center[i] = m_spacing[i] * (m_dim[i] - 1) * 0.5;
        center[i] += m_origin[i];
    }
}

}










