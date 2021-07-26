#pragma once
#include "DSObject.h"
#include "Bitvector.h"
#include "Grid.h"

namespace sjDS {

/**
@Class BitImage:
            An image (1D/2D/3D) that can store one bit per grid point.
            This representation is useful for storing boolean state of grid points
            (for e.g. selection state true/false) in an efficient manner.
*/
class BitImage : public DSObject
{
public:
    /// Default Constructor.
    BitImage();

    /// Construct based on image grid.
    BitImage(const Grid& grid);

    /// Reinitialize / resize bit-image. All previous data will be lost.
    int Resize(const Grid& grid);

    /// Set a grid-point bit to 1.
    int Set(const size_t p[3]);

    /// Set a grid-point bit to 0.
    int Clear(const size_t p[3]);

    /// Get the value of bit at a grid-point.
    /// return value:  0 --> bit value is 0
    /// return value:  1 --> bit value is 1
    /// return value: -1 --> Error. Invalid position of gridpoint.
    int Get(const size_t p[3]) const;

    /// Get value based on voxelId on logical-grid.
    //int Get(size_t voxelId) const;

    const void* GetRawPointer() const;
    const void* GetRawPointerToSlice(size_t zSliceNumber) const;

    void GetDimensions(size_t dim[3]) const;
    void GetByteDimensions(size_t dim[3]) const;

    /// Set all grid-point bits to 0.
     int ClearAll();
    /// Set all grid-point bits to 1.
    // int SetAll();

private:
    /// Allocate memory for internal bitvector based on the byte-dimensions of the grid.
    int _AllocateMemory();

    /// Validate a 3d-index based on logical grid.
    bool _ValidateIndex(const size_t ijk[3]) const;

    /// Calculate grid point position in 1D data array based on the dimensions of the physical grid.
    size_t _CalcVoxelID(const size_t ijk[3]) const;

    /// Member Variables
    Grid m_LogicalGrid;    /**< Logical grid defining the image. */
    Grid m_PhysicalGrid;   /**< Actual grid in memory based on padded bits. */
    Bitvector m_BitData;   /**< one dimentional bitvector to store all bits. */
};

}