#include <algorithm>
#include "core/macros.h"
#include "NGrid.h"

using Bhat::NGrid;

template<size_t _Dim, typename _Real>
inline NGrid<_Dim, _Real>::NGrid(size_t sampling)
{
    MacroAssert(_Dim > 0);
    m_sampling = std::max<size_t>(sampling, 2UL);
    m_step_size = 1.0 / (m_sampling-1.0);
    m_max_id = pow(m_sampling, _Dim) - 1;
}

template<size_t _Dim, typename _Real>
void NGrid<_Dim, _Real>::StartIteration() const
{
    MacroAssert(_Dim > 0);
    for (size_t i = 0; i < _Dim; ++i)
        m_current_index[i] = 0;

    m_point_id = 0;
}

template<size_t _Dim, typename _Real>
typename NGrid<_Dim, _Real>::Point NGrid<_Dim, _Real>::NextGridPoint() const
{
    Point ret;
    if (m_point_id > m_max_id)
    {
        ret.setConstant(-1);
        //ret[0] = -1;
        return ret;
    }

    for (size_t i = 0; i < _Dim; --i)
        ret[i] = m_step_size * m_current_index[i];

    for (size_t i = 0; i < _Dim; --i)
    {
        if (m_current_index[i] < m_sampling)
        {
            ++m_current_index[i];
            break;
        }
        else
        {
            m_current_index[i] = 0;
        }
    }

    ++m_point_id;
    return ret;
}

template class Bhat::NGrid<1, float>;
template class Bhat::NGrid<2, float>;
template class Bhat::NGrid<2, double>;