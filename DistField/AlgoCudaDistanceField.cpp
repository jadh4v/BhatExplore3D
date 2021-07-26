#include <vtkImageData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include "AlgoCudaDistanceField.h"
#include "core/macros.h"
#include "ds/Grid.h"
#include "ds/GridPoint.h"

int CUDAWrapper_ComputeDistanceField(const std::vector<float>& h_A, const std::vector<float>& h_B, std::vector<float>& h_C);
using voldef::AlgoCudaDistanceField;

AlgoCudaDistanceField::AlgoCudaDistanceField()
{
}

AlgoCudaDistanceField::AlgoCudaDistanceField(vtkImageData* imageMask, bool signedFunction)
{
    MacroAssert(imageMask != nullptr);
    //MacroAssert(imageMask->GetScalarType() == VTK_UNSIGNED_INT);
    auto dataArray = imageMask->GetPointData()->GetScalars();
    MacroAssert(dataArray != nullptr);
    vtkIdType gridPointsCount = imageMask->GetNumberOfPoints();
    std::vector<vtkIdType> domainPointIds;
    std::vector<double> domainPoints;
    std::vector<double> objectPoints;
    domainPointIds.reserve(gridPointsCount);
    domainPoints.reserve(gridPointsCount);
    objectPoints.reserve(gridPointsCount);
    sjDS::Grid grid;
    grid.SetDimensions(imageMask->GetDimensions());
    double p[3] = { 0,0,0 };
    for (vtkIdType pId = 0; pId < gridPointsCount; ++pId)
    {
        double maskValue = dataArray->GetVariantValue(pId).ToDouble();
        if (maskValue > 0)
        {
            sjDS::GridPoint g(pId, &grid);
            g.SetModeToAllNeighbors();
            g.StartNeighborIteration();
            unsigned int g_id = g.GetNextNeighborID();
            while (g_id != sjDS::GridPoint::cInvalidID)
            {
                double neighborValue = dataArray->GetVariantValue(g_id).ToDouble();
                if (neighborValue < 0.5)
                {
                    imageMask->GetPoint(g_id, p);
                    objectPoints.push_back(p[0]);
                    objectPoints.push_back(p[1]);
                    objectPoints.push_back(p[2]);
                }
                g_id = g.GetNextNeighborID();
            }
        }
    }

    // Identify domain points (points on which distance field will be computed).
    for (vtkIdType pId = 0; pId < gridPointsCount; ++pId)
    {
        imageMask->GetPoint(pId, p);
        domainPointIds.push_back(pId);
        domainPoints.push_back(p[0]);
        domainPoints.push_back(p[1]);
        domainPoints.push_back(p[2]);
    }

    this->SetDomainPoints(domainPoints.data(), domainPoints.size());
    this->SetObjectPoints(objectPoints.data(), objectPoints.size());
    m_SignedFunction = signedFunction;
    m_mask = imageMask;
}

AlgoCudaDistanceField::~AlgoCudaDistanceField()
{
}
 
template<typename T>
void AlgoCudaDistanceField::SetDomainPoints(const T* fieldDomainPoints, size_t numberOfValues)
{
    m_DomainPoints.resize(numberOfValues);
    for (size_t i = 0; i < numberOfValues; ++i)
        m_DomainPoints[i] = (float)fieldDomainPoints[i];
}

template<typename T>
void AlgoCudaDistanceField::SetObjectPoints(const T* objectPoints, size_t numberOfValues)
{
    m_ObjectPoints.resize(numberOfValues);
    for (size_t i = 0; i < numberOfValues; ++i)
        m_ObjectPoints[i] = (float)objectPoints[i];
}

int AlgoCudaDistanceField::input_validation() const
{
    if (m_DomainPoints.empty() || m_ObjectPoints.empty())
        return 0;

    if (m_DomainPoints.size() % 3 != 0 || m_ObjectPoints.size() % 3 != 0)
        return 0;

    return 1;
}

int AlgoCudaDistanceField::primary_process()
{
    // Allocate space for output distance values.
    // Since distance values are scalars they count will be 3 times less:
    m_OutputDistanceField.resize(m_DomainPoints.size() / 3);
    return CUDAWrapper_ComputeDistanceField(m_DomainPoints, m_ObjectPoints, m_OutputDistanceField);
}

int AlgoCudaDistanceField::post_processing()
{
    if(m_SignedFunction)
    {
        MacroAssert(m_mask != nullptr);
        auto dataArray = m_mask->GetPointData()->GetScalars();
        MacroAssert(dataArray != nullptr);
        for (vtkIdType i = 0; i < m_mask->GetNumberOfPoints(); ++i)
        {
            double maskValue = dataArray->GetVariantValue(i).ToDouble();
            if (maskValue > 0.1)
                m_OutputDistanceField[i] *= -1.0f;
        }
    }
    return 1;
}

template void AlgoCudaDistanceField::SetDomainPoints(const float*, size_t);
template void AlgoCudaDistanceField::SetDomainPoints(const double*, size_t);
template void AlgoCudaDistanceField::SetObjectPoints(const float*, size_t);
template void AlgoCudaDistanceField::SetObjectPoints(const double*, size_t);
