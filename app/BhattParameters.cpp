#include "BhattParameters.h"

void BhattParameters::SetParameters(
    double alpha,
    double stepSize,
    double del,
    double narrow_band,
    int num_of_iterations,
    int recomputePhiIterations,
    int downSampleFactor,
    int attribDim)
{
    this->m_Alpha = alpha;
    this->m_StepSize = stepSize;
    this->m_Del = del;
    this->m_NarrowBand = narrow_band;
    this->m_NumOfIterations = num_of_iterations;
    this->m_RecomputePhiIterations = recomputePhiIterations;
    this->m_DownSampleFactor = downSampleFactor;
    this->m_AttributeDimensions = attribDim;
}

void BhattParameters::SetGlobalRange(const double r[2])
{
    m_GlobalRange[0] = r[0];
    m_GlobalRange[1] = r[1];
}
