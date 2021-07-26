#include "AbstractRegion.h"

using Bhat::AbstractRegion; 

std::vector<double> AbstractRegion::IndicesToPoints(const std::set<vtkIdType>& indices) const
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

std::vector<double> AbstractRegion::IndicesToPoints(const std::vector<vtkIdType>& indices) const
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
