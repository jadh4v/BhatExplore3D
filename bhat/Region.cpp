#include <algorithm>
#include <core/macros.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <ds/Bitvector.h>
#include <ds/GridPoint.h>
#include "Region.h"

using Bhat::Region;

Region::Region(vtkImageData* regionMask)
{
    MacroAssert(regionMask);

    // Create base grid
    m_Grid.SetDimensions(regionMask->GetDimensions());
    m_Grid.SetOrigin(regionMask->GetOrigin());
    m_Grid.SetSpacing(regionMask->GetSpacing());

    // Create region using set of voxel Ids.
    auto data = regionMask->GetPointData()->GetScalars();
    double range[2];
    data->GetRange(range);
    for (vtkIdType i = 0; i < data->GetNumberOfValues(); ++i)
    {
        auto var = data->GetVariantValue(i);
        double value = data->GetVariantValue(i).ToDouble();
        if (value > 0.1)
            m_RegionVoxelIds.push_back(i);
    }
    std::sort(m_RegionVoxelIds.begin(), m_RegionVoxelIds.end());
}

std::set<vtkIdType> Region::BoundaryIndices() const
{
    std::set<vtkIdType> ret;
    double p[3] = { 0,0,0 };

    // For each voxel in the region, check if all its neighbors are also within the region.
    // If at least one neighbour is outside, then the current voxel is a boundary voxel.
    for (const auto& vId : m_RegionVoxelIds)
    {
        sjDS::GridPoint g(vId, &m_Grid);
        g.SetModeToAllNeighbors();
        g.StartNeighborIteration();
        vtkIdType nei_id = g.GetNextNeighborID();
        while (nei_id != sjDS::GridPoint::cInvalidID)
        {
            if ( !this->Contains(nei_id) )
            {
                ret.insert(vId);
                break;
            }
            nei_id = g.GetNextNeighborID();
        }
    }
    return ret;
}

std::vector<double> Region::BoundaryPoints() const
{
    std::vector<double> ret;

    // Get boundary indices and convert them into grid points.
    auto indices = BoundaryIndices();
    return IndicesToPoints(indices);
}

std::vector<double> Region::SubRegionBoundaryPoints(const sjDS::Bitvector& subRegion) const
{
    std::set<vtkIdType> ret_indices;
    for (size_t i = 0; i < m_RegionVoxelIds.size(); ++i)
    {
        if (subRegion.Get(i))
        {
            auto vId = m_RegionVoxelIds[i];
            sjDS::GridPoint g(vId, &m_Grid);
            g.SetModeToAllNeighbors();
            g.StartNeighborIteration();
            vtkIdType nei_id = g.GetNextNeighborID();
            while (nei_id != sjDS::GridPoint::cInvalidID)
            {
                auto pos = this->Find(nei_id);
                if (pos >= 0 && !subRegion.Get(pos))
                {
                    ret_indices.insert(vId);
                    break;
                }
                nei_id = g.GetNextNeighborID();
            }
        }
    }

    return IndicesToPoints(ret_indices);
}

std::vector<double> Region::Points() const
{
    return IndicesToPoints(m_RegionVoxelIds);
}

bool Region::Contains(vtkIdType v) const
{
    return std::binary_search(m_RegionVoxelIds.begin(), m_RegionVoxelIds.end(), v);
}

/*
template<class Container>
std::vector<double> Region::IndicesToPoints(const Container& indices) const
{
    std::vector<double> ret;
    ret.reserve(indices.size() * 3);
    double p[3] = { 0,0,0 };
    for (const auto& vId : indices)
    {
        m_Grid.GetPoint(vId, p);
        ret.push_back(p[0]);
        ret.push_back(p[1]);
        ret.push_back(p[2]);
    }
    return ret;
}
*/

vtkIdType Region::operator[](size_t i) const
{
    return m_RegionVoxelIds[i];
}

ptrdiff_t Region::Find(vtkIdType v) const
{
    auto lower = std::lower_bound(m_RegionVoxelIds.begin(), m_RegionVoxelIds.end(), v);
    if (lower != m_RegionVoxelIds.end() && *lower == v)
        return (lower - m_RegionVoxelIds.begin());
    else
        return -1;
}
