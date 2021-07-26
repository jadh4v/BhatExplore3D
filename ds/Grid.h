#ifndef GRID_H
#define GRID_H

#include <cstdint>
#include <cstddef>
#include "DSObject.h"

class QDataStream;
class vtkImageData;

namespace sjDS{

class Grid : public DSObject
{
public:
typedef uint32_t type_uid;

    Grid();
    Grid( const size_t dim[3] );
    Grid( vtkImageData* img );
    Grid( QDataStream& in_stream );

    size_t x() const { return m_dim[0]; }
    size_t y() const { return m_dim[1]; }
    size_t z() const { return m_dim[2]; }

    /// Get size of one-dimensional array required to store all elements of the grid.
    /// Equivalent to getting the number of points / voxels in the grid.
    size_t GetArraySize() const;

    /// Get grid dimensions (in terms of 3D array).
    void GetDimensions( size_t dim[3] ) const;
    void GetDimensions( int dim[3] ) const;

    /// Set grid dimensions (in terms of 3D array).
    void SetDimensions( const int dim[3] );
    void SetDimensions( const size_t dim[3] );
    void SetDimensions( size_t x, size_t y, size_t z );

    /// Set voxel dimensions.
    void SetSpacing(double x, double y, double z);
    void SetSpacing( const double spacing[3] );
    void GetSpacing( double spacing[3] ) const { spacing[0] = m_spacing[0]; spacing[1] = m_spacing[1]; spacing[2] = m_spacing[2]; }

    /// Set grid origin.
    void SetOrigin(double x, double y, double z);
    void SetOrigin( const double origin[3] );
    void GetOrigin( double origin[3] ) const { origin[0] = m_origin[0]; origin[1] = m_origin[1]; origin[2] = m_origin[2]; }

    void GetPoint( size_t ijk[3], double x[3]) const;

    void GetPoint( type_uid voxel_id, double x[3]) const;

    /// Calculate the center point of the grid bounds.
    void GetCenter(double center[3]) const;

    /// Access grid origin.
    const double* Origin() const  { return m_origin;  }
    const double* Spacing() const { return m_spacing; }
    const size_t* Dim() const { return m_dim; }

    /// Calculated the voxel id based on a given grid index.
    /// This is equivalent to the index of that voxel in a one-dimensional array.
    type_uid  CalcVoxelID( const size_t* ijk ) const;
    type_uid  CalcVoxelID( size_t i, size_t j, size_t k ) const;

    /// calculate grid index based on a given voxel id.
    int CalcGridIndex( const type_uid voxel_id, size_t ijk[3] ) const;

    /// convert ijk coordinates of the grid to normalized coordinates (typically used for texture interpolation).
    int IJKToNormalized(const size_t ijk[3], double image_normalized_coord[3]) const;
    int IJKToNormalized(size_t i, size_t j, size_t k, double image_normalized_coord[3]) const;

    /// convert normalized coordinates to ijk (floor).
    /// return parameter cell_normalized_residue[3] stores the remainder of the normalized coordinates, re-normalized at cell level.
    int NormalizedToIJK(const double image_normalized_coord[3], size_t ijk[3], double cell_normalized_residue[3]) const;

    /// Validate a given grid index.
    bool ValidateIndex( const size_t ijk[3] ) const;
    bool ValidateIndex( const int64_t ijk[3] ) const;
    bool ValidateIndex( size_t i, size_t j, size_t k ) const;

    bool is2D() const;
    bool is3D() const;

    int halve();
    int ScaleUpBy2();

    int Write(QDataStream& outstream) const;

    bool SameDimensions(const Grid& A) const;
    bool operator==(const Grid& A) const;
    bool operator!=(const Grid& A) const;


private:

    void init();

    int read(QDataStream& outstream);

    void recalculate_array_size();

    //-------------------
    size_t m_array_size;
    size_t m_dim[3];
    double m_spacing[3];
    double m_origin[3];
    //-------------------

};

}
#endif // GRID_H
