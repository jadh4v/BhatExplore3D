#pragma once

#include <vector>
#include <Eigen/Core>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <ds/Bitvector.h>

namespace Bhat {

template<size_t _Dim=2, typename _Real=float>
class FeatureSpace
{
public:
    typedef Eigen::Matrix<_Real, _Dim, 1> Point;

    /// Construct an empty feature space.
    FeatureSpace();

    /// Construct feature space from given points / point-features.
    FeatureSpace(const std::vector<Point>& voxel_features, const sjDS::Bitvector& mask);

    /// Push a point in the feature space.
    void Push(Point p);

    /// Update segmentation mask
    void UpdateMask(const sjDS::Bitvector& mask) { m_mask = mask; }

    /// Calculate approximate kernel bandwidth (sigma) for all dimensions of the feature space.
    void BuildKernel();

    /// Compute variance for a component of feature points.
    double GetVariance(size_t dimension) const;

    /// Compute Kernel-based density estimation for each of the regions.
    void ComputeP1();
    void ComputeP2();

    /// Compute function L
    void ComputeL();

    /// Compute function V
    void ComputeV(std::vector<_Real>& V, double del, double pi) const;

private:
    /// Validate if passed dim is valid.
    bool _ValidDimension(size_t d) const;

    /// normalize the feature space.
    void _Normalize(std::vector<Point>& points);

    /// Compute the kernel function value based on individual dimension bandwidths, given (z - z_i).
    double _Kernel(Point d) const;

    /// Compute second term of V(x)
    double _Compute_V_1st_Term() const;



    const double eps = 1e-16;
    std::vector<Point> m_points;    /// points in feature space
    std::vector<_Real> m_sigma;   /// kernel bandwidths
    sjDS::Bitvector m_mask;         /// segmentation mask (separation of background-foreground voxels).
    size_t m_modelSampling = 256;    /// sampling point-per-dimension for functions modelled over the feature-space.
    std::vector<_Real> m_Pout, m_Pin, m_L;
    _Real m_Ain = 0, m_Aout = 0;
};

}
