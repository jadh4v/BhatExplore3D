#pragma once
#include <set>
#include <vector>
#include <vtkType.h>
#include <ds/Grid.h>

class vtkImageData;
namespace sjDS {
    class Bitvector;
}

namespace Bhat {

class AbstractRegion
{
public:
    //AbstractRegion(vtkImageData* regionMask);
    //const std::vector<vtkIdType>& RegionVoxels() const { return m_RegionVoxelIds; }

    /// Get all voxels as grid points.
    virtual std::vector<double> Points() const = 0;

    /// Get Boundary voxel indices.
    /// Boundary is defined as the voxels inside the region that have at least one immediate neighbor that is outside.
    virtual std::set<vtkIdType> BoundaryIndices() const = 0;

    /// Get Boundary voxels as grid points.
    /// Boundary is defined as the voxels inside the region that have at least one immediate neighbor that is outside.
    virtual std::vector<double> BoundaryPoints() const = 0;

    /// Get Boundary points of a sub-region.
    virtual std::vector<double> SubRegionBoundaryPoints(const sjDS::Bitvector& subRegion) const = 0;

    /// Test if a given voxel Id is inside the region.
    virtual bool Contains(vtkIdType v) const = 0;

    /// Get size of region.
    virtual size_t Size() const = 0;

    /// Get voxelId in region using subscript - random access.
    virtual vtkIdType operator[](size_t i) const = 0;

protected:
    // Image grid over with the region is defined.
    sjDS::Grid m_Grid;
    std::vector<double> IndicesToPoints(const std::set<vtkIdType>& indices) const;
    std::vector<double> IndicesToPoints(const std::vector<vtkIdType>& indices) const;
};

}
