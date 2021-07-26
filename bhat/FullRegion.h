#pragma once
#include <set>
#include <cstddef>
#include <bhat/AbstractRegion.h>

class vtkImageData;
namespace sjDS {
    class Bitvector;
}

namespace Bhat {

class FullRegion : public AbstractRegion
{
public:
    FullRegion(vtkImageData* regionMask);

    /// Get all voxels as grid points.
    std::vector<double> Points() const;

    /// Get Boundary voxel indices.
    /// Boundary is defined as the voxels inside the region that have at least one immediate neighbor that is outside.
    std::set<vtkIdType> BoundaryIndices() const;

    /// Get Boundary voxels as grid points.
    /// Boundary is defined as the voxels inside the region that have at least one immediate neighbor that is outside.
    std::vector<double> BoundaryPoints() const;

    /// Get Boundary points of a sub-region.
    std::vector<double> SubRegionBoundaryPoints(const sjDS::Bitvector& subRegion) const;

    /// Test if a given voxel Id is inside the region.
    bool Contains(vtkIdType v) const;

    size_t Size() const { return m_Grid.GetArraySize(); }

    vtkIdType operator[](size_t i) const;
};

}
