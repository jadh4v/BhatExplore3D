#pragma once
#include <vector>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include "core/Algorithm.h"

namespace voldef {

/**
 Compute a distance field over a set of points that sample the a domain (FieldDomainPoints).
 Distances are computed based on provided Object (ObjectPoints).
 Essentially takes two sets of points as input and computes shortest distance from points
 of the first set to the points of the second set.
*/
class AlgoCudaDistanceField : public sjCore::Algorithm
{
public:
    AlgoCudaDistanceField();
    AlgoCudaDistanceField(vtkImageData* imageMask, bool signedFunction);
    virtual ~AlgoCudaDistanceField();

    // Input
    template<typename T>
    void SetDomainPoints(const T* fieldDomainPoints, size_t numberOfPoints);
    template<typename T>
    void SetObjectPoints(const T* objectPoints, size_t numberOfPoints);

    // Output
    std::vector<float> GetOutput() const { return m_OutputDistanceField; }

private:
    virtual int input_validation() const;
    virtual int primary_process();
    virtual int post_processing();

    //member variables
    std::vector<float> m_DomainPoints;
    std::vector<float> m_ObjectPoints;
    std::vector<float> m_OutputDistanceField;
    bool m_SignedFunction = false;
    vtkSmartPointer<vtkImageData> m_mask = nullptr;
};
}