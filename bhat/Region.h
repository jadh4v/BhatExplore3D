#pragma once
#include <set>
#include <cstddef>
#include <ds/Grid.h>
#include <bhat/AbstractRegion.h>

class vtkImageData;
namespace sjDS {
    class Bitvector;
}

namespace Bhat {

class Region : public AbstractRegion
{
public:
    Region(vtkImageData* regionMask);
    //const std::vector<vtkIdType>& RegionVoxels() const { return m_RegionVoxelIds; }

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

    size_t Size() const { return m_RegionVoxelIds.size(); }

    vtkIdType operator[](size_t i) const;

private:
    //std::vector<vtkIdType>::const_iterator Find(vtkIdType v) const;
    ptrdiff_t Find(vtkIdType v) const;

    //template<class Container>
    //std::vector<double> IndicesToPoints(const Container& indices) const;

    // Image grid over with the region is defined.
    //sjDS::Grid m_Grid;
    // Voxels in the grid that make the region.
    // convert this to CompressedSegment later.
    std::vector<vtkIdType> m_RegionVoxelIds;
};

}
