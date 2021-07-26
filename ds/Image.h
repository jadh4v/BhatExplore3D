#ifndef IMAGE_H
#define IMAGE_H

#include <cstdint>
#include <vector>
#include <QString>
#include <QDataStream>
//#include "graphseg_globals.h"
#include "core/macros.h"
#include "DSObject.h"

class vtkImageData;
class QImage;
typedef unsigned int type_uid;
typedef unsigned int type_uint;

namespace sjDS {

class VoxelRow;
class Grid;
class ImageSegmentation;


class Image : public DSObject
{
public:
    // Default Constructor
    Image();
    // Destructor
    virtual ~Image() noexcept;
    // Copy Constructor
    Image( const Image& img );
    // Copy Assignment Operator
    Image& operator=(const Image& A );
    // Move Constructor
    Image( Image&& img ) noexcept;
    // Move Assignment Operator
    Image& operator=(Image&& A ) noexcept;

    // Other Constructors
    explicit Image( vtkImageData* img );
    explicit Image( const QImage* img );
    Image( const size_t dim[3] );
    Image(size_t x, size_t y, size_t z);
    Image( const Grid* grid );
    Image( const type_uint* data, const Grid* grid );
    Image( const type_uint* data, const size_t dim[3] );
    /// Construct by reading a data stream / file.
    Image( QDataStream& in_stream );

    type_uint GetVoxel(type_uid voxel_id) const;
    type_uint GetVoxel(size_t ijk[3]) const;
    int SetVoxel(type_uid voxel_id, type_uint value);
    int SetVoxel(size_t ijk[3], type_uint value);
    int SetVoxel(size_t i, size_t j, size_t k, type_uint value);
    int SetVoxelRow(const sjDS::VoxelRow& row, type_uint value);
    int GetScalarRange(type_uint range[2]);
    int GetScalarRange(type_uint range[2]) const;
    int GetScalarRange(type_uint range[2], const std::vector<sjDS::VoxelRow>& region) const;
    void RecomputeScalarRange() const;
    void SetOrigin(double x, double y, double z);

    const type_uint* GetDataPointer() const { return m_griddata; }
          type_uint* GetDataPointer()       { return m_griddata; }

    void GetUniqueValues(std::vector<type_uint>& unique_values ) const;

    void ClearData();

    type_uint& operator[](size_t i) { return m_griddata[i]; }
    const type_uint& operator[](size_t i) const { return m_griddata[i]; }

    // Class Injection --> hseg::Grid
    MacroGetMember(const Grid*, m_grid, Grid)
    size_t GetArraySize() const;
    void GetDimensions( size_t dim[3] ) const;
    bool is2D() const;
    bool is3D() const;
    bool isSameDim(const Image&A) const;
    size_t x() const;
    size_t y() const;
    size_t z() const;

    /// Write the image to a file.
    int Write(const QString& filename) const;
    int Write(QDataStream& out_stream) const;
    ImageSegmentation* ConvertToImageSegmentation() const;
    void Threshold(type_uint minimum, type_uint maximum);
    void GetBounds(double bounds[6]) const;

private:

    void init();

    int allocate_data_array();

    int calculate_scalar_range(type_uint range[2]) const;

    int read_scalar_range(type_uint range[2]) const;

    /// Read image segmentation from a data stream.
    int read(QDataStream& in_stream);


    //----------------------
    Grid* m_grid;
    type_uint*  m_griddata;
    mutable type_uint   m_range[2];
    //----------------------

};

}
#endif // IMAGE_H
