#include <algorithm>
#include <core/macros.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <ds/Bitvector.h>
#include <ds/Grid.h>
#include <ds/GridPoint.h>
#include "FullRegion.h"

using Bhat::FullRegion;

FullRegion::FullRegion(vtkImageData* regionMask)
{
    MacroAssert(regionMask);

    // Create base grid
    this->m_Grid.SetDimensions(regionMask->GetDimensions());
    this->m_Grid.SetOrigin(regionMask->GetOrigin());
    this->m_Grid.SetSpacing(regionMask->GetSpacing());
}

std::set<vtkIdType> FullRegion::BoundaryIndices() const
{
    std::set<vtkIdType> ret;
    //for (auto k : {0, m_Grid.})
    return ret;
}

std::vector<double> FullRegion::BoundaryPoints() const
{
    std::vector<double> ret;
    // Get boundary indices and convert them into grid points.
    auto indices = BoundaryIndices();
    return IndicesToPoints(indices);
}

std::vector<double> FullRegion::SubRegionBoundaryPoints(const sjDS::Bitvector& subRegion) const
{
    std::set<vtkIdType> ret_indices;
    std::vector<double> ret_empty;
    MacroConfirmOrReturn((subRegion.Size() == m_Grid.GetArraySize()), ret_empty);

    for (size_t i = 0; i < m_Grid.GetArraySize(); ++i)
    {
        if (subRegion.Get(i))
        {
            vtkIdType vId = vtkIdType(i);
            sjDS::GridPoint g(vId, &m_Grid);
            g.SetModeToAllNeighbors();
            g.StartNeighborIteration();
            vtkIdType nei_id = g.GetNextNeighborID();
            while (nei_id != sjDS::GridPoint::cInvalidID)
            {
                if (!subRegion.Get(nei_id))
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

std::vector<double> FullRegion::Points() const
{
    double p[3] = { 0,0,0 };

    std::vector<double> ret;
    size_t numPts = m_Grid.GetArraySize();
    ret.reserve(numPts * 3);

    for (size_t i=0; i < numPts; ++i)
    {
        m_Grid.GetPoint(i, p);
        ret.push_back(p[0]);
        ret.push_back(p[1]);
        ret.push_back(p[2]);
    }

    return ret;
}

bool FullRegion::Contains(vtkIdType v) const
{
    return (v < m_Grid.GetArraySize());
}

vtkIdType FullRegion::operator[](size_t i) const
{
    //return m_RegionVoxelIds[i];
    if (i < m_Grid.GetArraySize())
        return i;
    else
        return -1;
}
