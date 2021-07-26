#pragma once
//#include "Vec.h"
#include <Eigen/Core>

namespace Bhat
{
template<size_t _Dim=1, typename _Real=float>
class NGrid
{
public:
    typedef Eigen::Matrix<_Real, _Dim, 1> Point;
    explicit NGrid(size_t sampling);
    void StartIteration() const;
    Point NextGridPoint() const;
    size_t NumberOfPoints() const { return (m_max_id + 1); }

private:
    size_t m_sampling = 16;
    size_t m_max_id = 0;
    double m_step_size = 0;

    // mutable members
    mutable size_t m_current_index[_Dim];
    mutable size_t m_point_id = 0;
};
}